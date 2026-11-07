"""IPOP-CMA-ES with adaptive restarts for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0
        d = self.dim
        budget = self.budget

        best_x = np.zeros(d)
        best_f = np.inf
        evals_used = 0

        def time_left():
            return timeout - (time.time() - start_time)

        def make_cma_state(m, sigma, lam):
            mu = max(1, lam // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))
            pc = np.zeros(d)
            C = np.eye(d)
            B = np.eye(d)
            D = np.ones(d)
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (32 * d * d))
            return dict(m=m, sigma=sigma, lam=lam, mu=mu, w=w, mu_eff=mu_eff,
                        c_sigma=c_sigma, c_c=c_c, c_1=c_1, c_mu=c_mu,
                        pc=pc, C=C, B=B, D=D, chi_d=chi_d)

        def cma_step(state, func, budget_remaining):
            nonlocal best_x, best_f, evals_used
            m = state['m'].copy()
            sigma = state['sigma']
            lam = state['lam']
            mu = state['mu']
            w = state['w']
            mu_eff = state['mu_eff']
            c_sigma = state['c_sigma']
            c_c = state['c_c']
            c_1 = state['c_1']
            c_mu = state['c_mu']
            pc = state['pc'].copy()
            C = state['C'].copy()
            B = state['B'].copy()
            D = state['D'].copy()
            chi_d = state['chi_d']

            if budget_remaining < lam:
                # (1+1)-ES steps
                for _ in range(budget_remaining):
                    if time_left() < 0.5:
                        break
                    y = m + sigma * np.random.randn(d)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                        m = y.copy()
                return False

            # Generate population
            X = np.array([m + sigma * B @ (D * np.random.randn(d)) for _ in range(lam)])
            X = np.clip(X, -5, 5)
            F = np.array([func(x) for x in X])
            evals_used += lam

            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()

            idx = np.argsort(F)
            x_sorted = X[idx]

            # Weighted mean
            m_new = sum(w[i] * x_sorted[i] for i in range(mu))

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((m_new - m) / sigma))

            # Heuristic sigma adaptation
            if np.linalg.norm(pc) / np.sqrt(max(1e-12, 1 - (1 - c_c) ** (2 * evals_used // lam))) < (1.4 + 2 / (d + 1)) * chi_d:
                sigma /= (1 + c_sigma)
            else:
                sigma *= (1 + c_sigma * 0.3)
                sigma *= np.exp(c_sigma * (np.linalg.norm(pc) / chi_d - 1))

            # Update C
            delta_h = 1.0 if np.linalg.norm(pc) / chi_d < (1.4 + 2 / (d + 1)) else 0.0
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)
            for i in range(mu):
                diff = x_sorted[i] - m_new
                C += c_mu * w[i] * np.outer(diff, diff)
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)
            except:
                B = np.eye(d)
                D = np.ones(d)
                C = np.eye(d)

            state['m'] = m_new
            state['sigma'] = sigma
            state['pc'] = pc
            state['C'] = C
            state['B'] = B
            state['D'] = D
            return True

        # IPOP-CMA-ES with restarts
        lam = 10
        min_lam = 4
        sigma_init = 2.0
        stagnation_limit = 3 * lam  # No improvement for this many evals -> restart
        last_improvement_eval = 0
        last_best_f = np.inf
        restart_count = 0
        max_restarts = 30

        # Initial diverse starts
        initial_centers = [
            np.zeros(d),
            np.array([3.0] * d),
            np.array([-3.0] * d),
        ]

        for center in initial_centers:
            if evals_used >= budget or time_left() < 1.0:
                break
            state = make_cma_state(center.copy(), sigma_init, lam)
            last_improvement_eval = evals_used
            last_best_f = best_f

            while evals_used < budget and time_left() > 1.0:
                if not cma_step(state, func, budget - evals_used):
                    break

                # Check for stagnation
                if best_f < last_best_f - 1e-10:
                    last_improvement_eval = evals_used
                    last_best_f = best_f

                if evals_used - last_improvement_eval >= stagnation_limit:
                    # Restart with reduced population
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
                    lam = max(min_lam, lam // 2)
                    sigma_init = max(0.1, sigma_init * 0.5)
                    # Restart from current best
                    state = make_cma_state(best_x.copy(), sigma_init, lam)
                    last_improvement_eval = evals_used
                    last_best_f = best_f
                    break
            else:
                break

        # Final (1+1)-ES polish
        if evals_used < budget and time_left() > 1.0:
            m = best_x.copy()
            sigma = 0.3
            while evals_used < budget and time_left() > 0.5:
                y = m + sigma * np.random.randn(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    if sigma < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
