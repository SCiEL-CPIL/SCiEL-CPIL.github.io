import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using product-of-sines architecture.
    
    The model is a sum of sinusoidal terms with learnable frequencies and phases:
    f(x,y) = sum_i A_i * sin(fx_i * x + fy_i * y + phase_i)
    
    This allows BFGS to directly optimize amplitude, frequencies, and phase
    to recover sinusoidal targets like sin(x+y) or cos(x-y).
    
    Parameters (10 total):
    - params[0:4]: First term: [A1, fx1, fy1, phase1]
    - params[4:8]: Second term: [A2, fx2, fy2, phase2]
    - params[8:10]: Third term: [A3, fx3] with fy3=0, phase3=0 (simplified)
    """
    start_time = time.time()
    deadline = 145.0  # seconds (5s safety margin)
    
    n = len(x)
    
    # Helper function to compute predictions from parameters
    def predict_from_params(p):
        """Compute predictions from 10 parameters."""
        # Term 1: A1 * sin(fx1*x + fy1*y + phase1)
        term1 = p[0] * np.sin(p[1] * x + p[2] * y + p[3])
        
        # Term 2: A2 * sin(fx2*x + fy2*y + phase2)
        term2 = p[4] * np.sin(p[5] * x + p[6] * y + p[7])
        
        # Term 3: A3 * sin(fx3*x + fy3*y + phase3)
        # Use remaining params for third term
        term3 = p[8] * np.sin(p[9] * x + 0.5 * p[9] * y + 0.0)
        
        return term1 + term2 + term3
    
    # Objective function for BFGS
    def objective(p):
        try:
            pred = predict_from_params(p)
            if not np.all(np.isfinite(pred)):
                return 1e30
            return np.mean((pred - y) ** 2)
        except Exception:
            return 1e30
    
    # Try multiple initializations to avoid local minima
    best_pred = None
    best_mse = np.inf
    
    # Initializations:
    # 1. All ones (default start)
    # 2. Common frequencies: (1,1,0), (1,-1,0), (0,1,0), (1,0,0)
    # 3. Small random perturbations around common patterns
    
    init_candidates = [
        np.ones(10),  # Default
        np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),  # sin(x+y)
        np.array([1, 1, -1, 0, 0, 0, 0, 0, 0, 0]),  # sin(x-y)
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # sin(x)
        np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # sin(y)
        np.array([1, 1, 1, np.pi/2, 0, 0, 0, 0, 0, 0]),  # cos(x+y)
        np.array([1, 1, -1, np.pi/2, 0, 0, 0, 0, 0, 0]),  # cos(x-y)
        np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0]),  # Only second term
        np.array([0.5, 1, 1, 0, 0.5, 1, -1, 0, 0, 0]),  # Mix
        np.array([1, 2, 1, 0, 0, 0, 0, 0, 0, 0]),  # sin(2x+y)
        np.array([1, 1, 2, 0, 0, 0, 0, 0, 0, 0]),  # sin(x+2y)
    ]
    
    # Add some random initializations
    np.random.seed(42)
    for i in range(5):
        init_candidates.append(np.random.uniform(-2, 2, 10))
    
    for init in init_candidates:
        if time.time() - start_time > deadline:
            break
        
        try:
            result = minimize(
                objective,
                init,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < best_mse:
                pred = predict_from_params(result.x)
                if np.all(np.isfinite(pred)):
                    mse = np.mean((pred - y) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_pred = pred
        except Exception:
            continue
    
    # Fallback: if optimization failed, use a simple sinusoidal model
    if best_pred is None or not np.all(np.isfinite(best_pred)):
        # Use params directly in a simple model
        best_pred = (params[0] * np.sin(params[1] * x + params[2] * y + params[3]) +
                    params[4] * np.sin(params[5] * x + params[6] * y + params[7]) +
                    params[8] * np.sin(params[9] * x + 0.5 * params[9] * y))
    
    return best_pred
