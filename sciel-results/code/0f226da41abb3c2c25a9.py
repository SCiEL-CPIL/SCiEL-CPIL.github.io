"""Symbolic regression with rich feature basis for 2-input functions."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0


def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_ok():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _SAFETY_MARGIN


def _build_features(x, y):
    """Build a diverse, bounded feature matrix from 2D inputs."""
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Clip to avoid numerical issues
    xc = np.clip(x, -1.0, 1.0)
    yc = np.clip(y, -1.0, 1.0)
    
    # Radial and angular features
    r = np.sqrt(xc * xc + yc * yc)
    theta = np.arctan2(yc, xc)
    
    # Build feature list - we'll select 10 features for the 10 params
    # Include a constant term and diverse basis functions
    features = []
    features.append(np.ones(n))                          # 0: constant
    features.append(xc)                                   # 1: x
    features.append(yc)                                   # 2: y
    features.append(xc * xc)                              # 3: x^2
    features.append(yc * yc)                              # 4: y^2
    features.append(xc * yc)                              # 5: x*y
    features.append(np.sin(xc))                           # 6: sin(x)
    features.append(np.sin(yc))                           # 7: sin(y)
    features.append(np.cos(xc * yc))                      # 8: cos(x*y)
    features.append(r)                                    # 9: radial distance
    
    # Stack into matrix (n, 10)
    F = np.column_stack(features)
    
    # Ensure finite values
    F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using 10 learned parameters."""
    _init_deadline()
    
    if not _time_ok():
        # Safety fallback: simple linear
        return params[0] * np.asarray(x) + params[1] * np.asarray(y) + params[2]
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if params.shape[0] < 10:
        # Pad params if needed
        p = np.ones(10, dtype=np.float64)
        p[:min(10, len(params))] = params[:min(10, len(params))]
        params = p
    
    F = _build_features(x, y)
    
    # Prediction: F @ params
    pred = F @ params[:10]
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
