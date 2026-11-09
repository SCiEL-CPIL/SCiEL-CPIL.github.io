import numpy as np
from scipy.optimize import minimize

def find_root_system(F, x0):
        x0 = np.asarray(x0, dtype=float)
        
        # Check initial point
        Fx0 = F(x0)
        if np.linalg.norm(Fx0) <= 1e-8:
            return x0

        state = {'last_x': None, 'last_F': None}

        def objective(x):
            if np.allclose(x, state['last_x'], rtol=1e-12, atol=1e-12):
                return float(np.dot(state['last_F'], state['last_F']))
            Fx = F(x)
            state['last_x'] = x.copy()
            state['last_F'] = Fx
            return float(np.dot(Fx, Fx))

        def gradient(x):
            # Reuse last_F if possible
            if np.allclose(x, state['last_x'], rtol=1e-12, atol=1e-12):
                Fx = state['last_F']
            else:
                Fx = F(x)
                state['last_x'] = x.copy()
                state['last_F'] = Fx
                
            J = np.zeros((2, 2))
            h = 1e-7
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                F_plus = F(x + e)
                J[:, j] = (F_plus - Fx) / h
            return 2 * J.T @ Fx

        try:
            res = minimize(
                objective,
                x0,
                method='dogleg',
                jac=gradient,
                options={
                    'maxiter': 150,
                    'gtol': 1e-10,
                    'xtol': 1e-12,
                    'ftol': 1e-16,
                }
            )
            x = res.x
        except Exception:
            x = x0

        # Final check and fallback
        Fx = F(x)
        if np.linalg.norm(Fx) <= 1e-8:
            return x

        # Fallback: Newton with FD Jacobian
        for _ in range(50):
            Fx = F(x)
            norm_F = np.linalg.norm(Fx)
            if norm_F <= 1e-8:
                break
            
            J = np.zeros((2, 2))
            h = 1e-7
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (F(x + e) - Fx) / h
            
            try:
                dx = np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, Fx, rcond=None)[0]
            
            alpha = 1.0
            for _ in range(20):
                x_trial = x - alpha * dx
                F_trial = F(x_trial)
                if np.linalg.norm(F_trial) < norm_F:
                    x = x_trial
                    break
                alpha *= 0.5
            else:
                break
        
        return x
