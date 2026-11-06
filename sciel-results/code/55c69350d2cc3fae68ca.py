import time
import zlib
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0  # Safety margin: stop 60s before 330s kill

# Pre-compute a high-quality reference cap set using greedy with a strong heuristic
def _build_reference_cap_set(n, max_vectors=500):
    """Build a reference cap set using greedy with a refined heuristic."""
    if n < 2 or n > 10:  # Only pre-compute for reasonable dimensions
        return set()
    
    # Generate all vectors of {1,2}^n (cap sets typically avoid 0s for larger sizes)
    # But we should consider {0,1,2}^n. However, 0s create many conflicts.
    # Let's try to build a set that is dense in {1,2}^n first, as that's often optimal.
    # Actually, the best cap sets in small dimensions often use all symbols.
    # But greedy on {1,2}^n is fast and yields good baselines.
    # Let's generate all vectors in {0,1,2}^n but prioritize those with fewer 0s.
    
    # For efficiency in pre-computation, we'll just sample or use a heuristic sort.
    # Since 3^n can be large, we limit n to 10 for pre-computation.
    
    # Heuristic: vectors with balanced 1s and 2s and few 0s are better.
    candidates = []
    for i in range(3 ** n):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        v = tuple(v)
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        # Score: prefer fewer 0s, balanced 1s and 2s
        score = -10 * c0 - abs(c1 - c2)
        candidates.append((score, v))
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    cap_set = set()
    for _, v in candidates:
        if len(cap_set) >= max_vectors:
            break
        # Check if adding v keeps it a cap set
        is_cap = True
        # Check against existing vectors
        # To speed up, we can check: for any a, b in cap_set, a+b+v != 0 mod 3
        # This is O(k^2) where k is current size. k <= 500, so 250k checks.
        # For n=10, this is manageable.
        for a in cap_set:
            for b in cap_set:
                if a == b:
                    continue
                conflict = True
                for k in range(n):
                    if (a[k] + b[k] + v[k]) % 3 != 0:
                        conflict = False
                        break
                if conflict:
                    is_cap = False
                    break
            if not is_cap:
                break
        
        # Also check pairs (a, a) + v? No, distinct vectors.
        # But we need to check if v forms a line with two *other* vectors.
        # The above loop checks all pairs (a,b) from cap_set.
        # If a+b+v = 0, then {a,b,v} is a line.
        # This covers all cases where v is the third point.
        
        if is_cap:
            cap_set.add(v)
    
    return cap_set

# Learn a surrogate model from the reference cap set
def _learn_surrogate(n, ref_cap_set):
    """Learn a fast surrogate that approximates marginal conflict cost."""
    if not ref_cap_set or len(ref_cap_set) < 10:
        return None
    
    # Extract features from vectors
    def _extract_features(v):
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        imbalance = abs(c1 - c2)
        # Positional features: which coordinates have 1s and 2s
        # Use a simple hash of positions to keep dimension low
        pos_1 = 0
        pos_2 = 0
        for i in range(n):
            if v[i] == 1:
                pos_1 |= (1 << i)
            elif v[i] == 2:
                pos_2 |= (1 << i)
        # Truncate to 16 bits to avoid huge dimensions
        pos_1 = pos_1 % 65536
        pos_2 = pos_2 % 65536
        return [c0, c1, c2, imbalance, pos_1, pos_2]
    
    # Create training data: in-cap-set vectors get high priority, others low
    in_set = set(ref_cap_set)
    all_vectors_sample = list(in_set)
    
    # Add some random out-of-set vectors
    import random
    random.seed(42)
    num_neg = min(500, 3 ** n - len(in_set))
    for _ in range(num_neg):
        v = tuple(random.randint(0, 2) for _ in range(n))
        if v not in in_set:
            all_vectors_sample.append(v)
    
    X = np.array([_extract_features(v) for v in all_vectors_sample], dtype=np.float64)
    y = np.array([1.0 if v in in_set else 0.0 for v in all_vectors_sample], dtype=np.float64)
    
    # Simple linear model with regularization
    if len(X) < 10:
        return None
    
    # Add bias term
    X_b = np.column_stack([X, np.ones(len(X))])
    
    # Regularized least squares
    lam = 1.0
    A = X_b.T @ X_b + lam * np.eye(X_b.shape[1])
    b = X_b.T @ y
    try:
        weights = np.linalg.solve(A, b)
    except:
        return None
    
    return weights

# Pre-compute surrogate at module load
_SURROGATE_WEIGHTS = {}
for _n in range(2, 11):
    try:
        _ref = _build_reference_cap_set(_n, max_vectors=300)
        if _ref:
            _weights = _learn_surrogate(_n, _ref)
            if _weights is not None:
                _SURROGATE_WEIGHTS[_n] = _weights
    except:
        pass

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def priority(vector, n):
    # Check deadline to avoid being killed mid-evaluation
    if time.time() > _DEADLINE:
        return -1e10  # Signal to stop adding vectors
    
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    h = _hash_vec(vector)
    
    # Use learned surrogate if available
    if n in _SURROGATE_WEIGHTS:
        weights = _SURROGATE_WEIGHTS[n]
        # Extract features
        imbalance = abs(c1 - c2)
        pos_1 = 0
        pos_2 = 0
        for i in range(n):
            if vector[i] == 1:
                pos_1 |= (1 << i)
            elif vector[i] == 2:
                pos_2 |= (1 << i)
        pos_1 = pos_1 % 65536
        pos_2 = pos_2 % 65536
        features = [c0, c1, c2, imbalance, pos_1, pos_2, 1.0]
        surrogate_score = sum(w * f for w, f in zip(weights, features))
        # Combine: surrogate gives learned preference, hash breaks ties
        # Scale surrogate to dominate hash for ordering
        return surrogate_score * 1000.0 + h / (3 ** n)
    
    # Fallback to original heuristic (zlib-based or simple)
    # Use the current program's logic as fallback if no surrogate
    s = ''.join(str(x) for x in vector)
    return zlib.crc32(s.encode('utf-8'))
