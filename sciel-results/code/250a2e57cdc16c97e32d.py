# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = None
_cache = {}

def _get_features(x, y):
    """
    Constructs a rich, diverse set of candidate features.
    Returns a dictionary of numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Use a robust key for caching
    # Since x and y are fixed during the optimization phase, we can cache based on shape and a hash of the first few elements
    # Using id is risky if arrays are copied, but the evaluator usually reuses buffers.
    # Let's use a hash of the flattened array for safety, but that might be slow.
    # Given 200 points, hashing is fast.
    try:
        key = (x.shape, y.shape, hash(tuple(x[:5])), hash(tuple(y[:5])))
    except:
        key = (x.shape, y.shape, id(x), id(y))
    
    if key in _cache:
        return _cache[key]
    
    # Clip for safety
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    
    # Feature Set
    # 1. Polynomial basis up to degree 4 (symmetric and mixed)
    # 2. Trigonometric basis with various frequencies
    # 3. Sigmoidal / Rational basis
    
    features = {}
    
    # Polynomials
    features['1'] = np.ones_like(x)
    features['x'] = x
    features['y'] = y
    features['x2'] = x**2
    features['y2'] = y**2
    features['xy'] = x * y
    features['x3'] = x**3
    features['y3'] = y**3
    features['x2y'] = x**2 * y
    features['xy2'] = x * y**2
    features['x4'] = x**4
    features['y4'] = y**4
    features['x2y2'] = x**2 * y**2
    
    # Trigonometric
    for k in [1, 2, 3, 5, 10]:
        features[f'sin{k}x'] = np.sin(k * x)
        features[f'sin{k}y'] = np.sin(k * y)
        features[f'cos{k}x'] = np.cos(k * x)
        features[f'cos{k}y'] = np.cos(k * y)
        features[f'sin{k}xy'] = np.sin(k * x * y)
        
    # Sigmoidal / Soft
    features['tanhx'] = np.tanh(x)
    features['tanhy'] = np.tanh(y)
    features['tanhxy'] = np.tanh(x * y)
    features['tanhx2'] = np.tanh(x**2)
    features['tanhy2'] = np.tanh(y**2)
    
    # Rational-like (smoothed)
    features['1/(1+x^2)'] = 1.0 / (1.0 + x**2)
    features['1/(1+y^2)'] = 1.0 / (1.0 + y**2)
    features['1/(1+x^2+y^2)'] = 1.0 / (1.0 + x**2 + y**2)
    
    # Store in cache
    _cache[key] = features
    return features

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check time limit
    if time.time() - _START_TIME > 145.0:
        # Return a safe fallback, e.g., mean of y
        return np.full_like(y, np.mean(y), dtype=float)
        
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if params.shape[0] != 10:
        # Pad or trim
        if params.shape[0] < 10:
            params = np.pad(params, (0, 10 - params.shape[0]))
        else:
            params = params[:10]
            
    # Get features
    feats = _get_features(x, y)
    
    # Select 10 features to use.
    # We want a diverse set.
    # Let's pick:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x2
    # 4: y2
    # 5: xy
    # 6: tanh(x)
    # 7: tanh(y)
    # 8: sin(2x)
    # 9: sin(2y)
    
    # This is a fixed selection, but the "strategy" is using a richer pool and
    # potentially allowing the optimizer to find better combinations if we were to
    # do feature selection. But here we stick to a fixed set of 10 from the pool.
    # To be more adaptive, we could try to use the params to *select* or *weight*
    # features in a non-linear way, but that's hard with BFGS.
    
    # Let's try a slightly different fixed set that includes some higher order terms
    # which the previous program might have missed or weighted poorly.
    
    # Actually, the previous program had:
    # p0: 1
    # p1: x
    # p2: y
    # p3: x2
    # p4: y2
    # p5: xy
    # p6: sin(p7*x + p8*y)
    # p7: freq1
    # p8: freq2
    # p9: sin(p7*y + p8*x)
    
    # This new strategy uses a FIXED basis but with more diverse non-polynomial terms
    # that don't require frequency parameters. This makes the optimization problem
    # LINEAR in the parameters, which is much easier for BFGS to solve (convex).
    # The previous program was non-linear in p7, p8, which could lead to local minima.
    
    # Let's use these 10 features:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x2
    # 4: y2
    # 5: xy
    # 6: tanh(x)
    # 7: tanh(y)
    # 8: sin(x)
    # 9: sin(y)
    
    f0 = feats['1']
    f1 = feats['x']
    f2 = feats['y']
    f3 = feats['x2']
    f4 = feats['y2']
    f5 = feats['xy']
    f6 = feats['tanhx']
    f7 = feats['tanhy']
    f8 = feats['sin1x']
    f9 = feats['sin1y']
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    pred = (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 + 
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.full_like(y, np.mean(y), dtype=float)
        
    return pred
# EVOLVE-BLOCK-END
