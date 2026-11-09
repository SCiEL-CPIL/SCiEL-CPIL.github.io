"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import UnivariateSpline


def fit(X_train, y_train):
    X = X_train[:, 0]  # 1D input
    y = y_train
    n = len(X)
    
    # Define candidate basis functions as callables
    def make_candidates(x):
        cands = []
        names = []
        
        # Polynomials
        for k in range(0, 9):
            cands.append(x ** k)
            names.append(f'x^{k}')
        
        # Trig
        cands.append(np.sin(x))
        names.append('sin(x)')
        cands.append(np.cos(x))
        names.append('cos(x)')
        cands.append(np.tanh(x))
        names.append('tanh(x)')
        cands.append(np.arctan(x))
        names.append('arctan(x)')
        
        # Exp and log
        cands.append(np.exp(x))
        names.append('exp(x)')
        cands.append(np.log(np.abs(x) + 1e-8))
        names.append('log(|x|)')
        cands.append(np.sqrt(np.abs(x)))
        names.append('sqrt(|x|)')
        
        # Abs and sign
        cands.append(np.abs(x))
        names.append('|x|')
        
        # Combinations
        cands.append(x * np.sin(x))
        names.append('x*sin(x)')
        cands.append(x * np.cos(x))
        names.append('x*cos(x)')
        cands.append(x * np.exp(x))
        names.append('x*exp(x)')
        cands.append(x * np.abs(x))
        names.append('x*|x|')
        cands.append(np.exp(x) * np.sin(x))
        names.append('exp(x)*sin(x)')
        cands.append(np.exp(x) * np.cos(x))
        names.append('exp(x)*cos(x)')
        cands.append(x ** 2 * np.sin(x))
        names.append('x^2*sin(x)')
        cands.append(x ** 2 * np.cos(x))
        names.append('x^2*cos(x)')
        cands.append(1.0 / (x + 1e-8))
        names.append('1/x')
        cands.append(1.0 / (x ** 2 + 1e-8))
        names.append('1/x^2')
        
        return cands, names
    
    train_cands, train_names = make_candidates(X)
    
    best_mse = np.inf
    best_feature_indices = None
    best_w = None
    best_type = None  # 'lsq' or 'spline' or 'interp'
    best_spline = None
    best_interp_X = None
    best_interp_y = None
    
    # Try single candidates
    for i in range(len(train_cands)):
        try:
            A = np.column_stack([np.ones(n), train_cands[i]])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ w
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_feature_indices = [i]
                best_w = w
                best_type = 'lsq'
        except:
            pass
    
    # Try pairs
    if best_mse > 1e-8:
        for i in range(len(train_cands)):
            for j in range(i + 1, len(train_cands)):
                try:
                    A = np.column_stack([np.ones(n), train_cands[i], train_cands[j]])
                    w, *_ = np.linalg.lstsq(A, y, rcond=None)
                    pred = A @ w
                    mse = np.mean((pred - y) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_feature_indices = [i, j]
                        best_w = w
                        best_type = 'lsq'
                except:
                    pass
    
    # Try triples
    if best_mse > 1e-8:
        for i in range(len(train_cands)):
            for j in range(i + 1, len(train_cands)):
                for k in range(j + 1, len(train_cands)):
                    try:
                        A = np.column_stack([np.ones(n), train_cands[i], train_cands[j], train_cands[k]])
                        w, *_ = np.linalg.lstsq(A, y, rcond=None)
                        pred = A @ w
                        mse = np.mean((pred - y) ** 2)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature_indices = [i, j, k]
                            best_w = w
                            best_type = 'lsq'
                    except:
                        pass
    
    # If no good LSQ model, try high-degree polynomial
    if best_mse > 1e-6:
        for deg in range(12, 2, -1):
            A = np.vander(X, N=deg + 1, increasing=True)
            try:
                w, *_ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ w
                mse = np.mean((pred - y) ** 2)
                if mse < 1e-6:
                    best_mse = mse
                    best_type = 'poly'
                    best_w = w
                    best_poly_deg = deg
                    break
            except:
                pass
    
    # If still not good, use spline
    if best_type is None or (best_type != 'lsq' and best_type != 'poly' and best_mse > 1e-4):
        try:
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
            spl = UnivariateSpline(X_sorted, y_sorted, s=0, k=3)
            best_type = 'spline'
            best_spline = spl
            best_mse = 0.0
        except:
            try:
                sort_idx = np.argsort(X)
                X_sorted = X[sort_idx]
                y_sorted = y[sort_idx]
                best_type = 'interp'
                best_interp_X = X_sorted
                best_interp_y = y_sorted
                best_mse = 0.0
            except:
                # Last resort
                best_type = 'const'
                best_w = np.array([np.mean(y)])
    
    # Create predict function
    if best_type == 'lsq':
        def predict(X_new):
            x = X_new[:, 0]
            cands, _ = make_candidates(x)
            A = np.column_stack([np.ones(len(x))] + [cands[i] for i in best_feature_indices])
            return A @ best_w
        return predict
    
    elif best_type == 'poly':
        def predict(X_new):
            x = X_new[:, 0]
            A = np.vander(x, N=best_poly_deg + 1, increasing=True)
            return A @ best_w
        return predict
    
    elif best_type == 'spline':
        def predict(X_new):
            x = X_new[:, 0]
            return best_spline(x)
        return predict
    
    elif best_type == 'interp':
        def predict(X_new):
            x = X_new[:, 0]
            return np.interp(x, best_interp_X, best_interp_y)
        return predict
    
    else:  # const
        def predict(X_new):
            return np.full(len(X_new), best_w[0])
        return predict

# EVOLVE-BLOCK-END
