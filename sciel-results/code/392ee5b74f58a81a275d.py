import numpy as np

class Optimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) Optimizer.
    Uses a population-based approach with adaptive step size and covariance matrix
    to minimize the black-box function in 5 dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 12
        mu = lam // 2
        
        # Initialize mean m at the center of the box
        m = np.zeros(d)
        # Initial step size sigma
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(d)
        
        # Eigen decomposition for efficient sampling
        D = np.ones(d)
        B = np.eye(d)
        
        # Paths for sigma adaptation
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Weights for the mean update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + 4)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff)))
        
        # Damping for sigma update
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Evaluation counter
        evals = 0
        best_x = m.copy()
        best_f = np.inf
        
        # Main loop
        while evals < budget:
            m_old = m.copy()
            
            # Sample population
            X = np.zeros((lam, d))
            for i in range(lam):
                z = np.random.randn(d)
                y = B @ (D * z)
                x = m + sigma * y
                X[i] = x
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if evals >= budget:
                    break
                x_eval = np.clip(X[i], -5, 5)
                F[i] = func(x_eval)
                evals += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_eval.copy()
            
            if evals >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean
            m = np.zeros(d)
            for i in range(mu):
                m += weights[i] * X[i]
            
            # Update covariance matrix and paths
            # Calculate weighted sum of outer products of successful steps
            Y = (X[:mu] - m_old) / sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (np.linalg.solve(C, (m - m_old))) / sigma
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals / lam)))) < (1.4 + 2 / (d + 1)) * chi_n
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (evals / lam)))) < (1.4 + 2 / (d + 1)) * chi_n
            
            # Update C
            delta_h = (1 - h_c) * c_c * (2 - c_c) * np.linalg.norm(p_c)**2
            RankMu = np.zeros((d, d))
            for i in range(mu):
                RankMu += weights[i] * np.outer(Y[i], Y[i])
            
            RankOne = np.outer(p_c, p_c) + delta_h
            
            C = (1 - c_1 - c_mu) * C + c_1 * RankOne + c_mu * RankMu
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
        return best_x, best_f
