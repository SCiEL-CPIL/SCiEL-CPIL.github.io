"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from functools import lru_cache

# EVOLVE-BLOCK-START
def _generate_all_vectors(n, limit=None):
    """Generate all vectors in {0,1,2}^n, optionally limited."""
    total = 3 ** n
    if limit is not None:
        total = min(total, limit)
    vectors = []
    for i in range(total):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        vectors.append(tuple(v))
    return vectors

def _is_cap_set(vecs):
    """Check if a set of vectors forms a cap set (no 3-term AP)."""
    if len(vecs) <= 2:
        return True
    vecs_arr = np.array(vecs, dtype=np.int8)
    n = len(vecs_arr)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check if vecs[i], vecs[j], vecs[k] form a 3-term AP
                # A 3-term AP means: vecs[i] + vecs[k] = 2 * vecs[j] mod 3
                # OR any permutation forms an AP
                vi, vj, vk = vecs_arr[i], vecs_arr[j], vecs_arr[k]
                if np.all((vi + vk) % 3 == (2 * vj) % 3):
                    return False
                if np.all((vi + vj) % 3 == (2 * vk) % 3):
                    return False
                if np.all((vj + vk) % 3 == (2 * vi) % 3):
                    return False
    return True

def _greedy_cap_set(n, num_candidates=2000, seed=42):
    """Build a cap set greedily using a good ordering heuristic."""
    rng = np.random.RandomState(seed)
    
    # Generate candidate vectors
    if n <= 6:
        candidates = _generate_all_vectors(n)
    else:
        candidates = [tuple(rng.randint(0, 3, n).tolist()) for _ in range(num_candidates)]
    
    # Sort by a heuristic: prefer balanced vectors with low local AP structure
    def heuristic(vec):
        v = np.array(vec, dtype=np.int8)
        c0 = np.sum(v == 0)
        c1 = np.sum(v == 1)
        c2 = np.sum(v == 2)
        target = n / 3.0
        balance = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
        
        ap_count = 0
        for i in range(n - 2):
            a, b, c = v[i], v[i+1], v[i+2]
            if (a + c) % 3 == (2 * b) % 3:
                ap_count += 1
        
        return -(balance + ap_count)
    
    candidates.sort(key=heuristic, reverse=True)
    
    # Greedily build cap set
    cap_set = []
    for vec in candidates:
        if len(cap_set) >= min(30, 3**n // 10):
            break
        test_set = cap_set + [vec]
        if _is_cap_set(test_set):
            cap_set.append(vec)
    
    return cap_set

def _compute_conflict_features(vec, cap_set):
    """Compute features for a vector relative to a cap set."""
    v = np.array(vec, dtype=np.int8)
    
    # Feature 1: Balance
    c0 = np.sum(v == 0)
    c1 = np.sum(v == 1)
    c2 = np.sum(v == 2)
    n = len(v)
    target = n / 3.0
    balance = [(c0 - target)**2, (c1 - target)**2, (c2 - target)**2]
    
    # Feature 2: Local AP count
    ap_count = 0
    for i in range(n - 2):
        a, b, c = v[i], v[i+1], v[i+2]
        if (a + c) % 3 == (2 * b) % 3:
            ap_count += 1
    
    # Feature 3: Distance to each vector in cap set
    distances = []
    for cv in cap_set:
        cv_arr = np.array(cv, dtype=np.int8)
        dist = np.sum(v != cv_arr)
        distances.append(dist)
    
    # Feature 4: For each vector in cap set, compute the "midpoint" check
    # If v and cv form an AP with some midpoint m, then m = (v + cv) / 2 mod 3
    # Check how many such midpoints are in the cap set
    midpoint_conflicts = 0
    cap_set_arr = np.array(cap_set, dtype=np.int8)
    for cv in cap_set:
        cv_arr = np.array(cv, dtype=np.int8)
        # Midpoint: (v + cv) / 2 mod 3, which is (v + cv) * 2^{-1} mod 3
        # 2^{-1} mod 3 = 2, so midpoint = (v + cv) * 2 mod 3
        mid = ((v + cv_arr) * 2) % 3
        # Check if mid is in cap_set
        for cv2 in cap_set_arr:
            if np.all(mid == cv2):
                midpoint_conflicts += 1
                break
    
    return np.array([c0, c1, c2, ap_count, midpoint_conflicts, len(cap_set)] + distances, dtype=np.float64)

def _train_mlp(X, y, hidden_size=32, epochs=500, lr=0.01, seed=42):
    """Train a simple MLP using numpy."""
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    
    # Initialize weights
    W1 = rng.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden_size)
    W2 = rng.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(1)
    
    for epoch in range(epochs):
        # Forward pass
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))  # Sigmoid
        
        # Loss
        eps = 1e-7
        loss = -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
        
        # Backward pass
        dz2 = (a2 - y) / n_samples
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1**2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    
    return W1, b1, W2, b2

def _mlp_predict(X, W1, b1, W2, b2):
    """Make predictions with the trained MLP."""
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    return 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))

@lru_cache(maxsize=10000)
def _get_trained_model(n):
    """Get or train the MLP for a given n."""
    # Build reference cap set
    cap_set = _greedy_cap_set(n, num_candidates=1000, seed=42)
    
    if len(cap_set) < 3:
        return None
    
    # Generate training data
    rng = np.random.RandomState(123)
    n_train = min(3**n, 2000)
    
    if n <= 6:
        all_vecs = _generate_all_vectors(n)
        if len(all_vecs) <= n_train:
            train_vecs = all_vecs
        else:
            indices = rng.choice(len(all_vecs), n_train, replace=False)
            train_vecs = [all_vecs[i] for i in indices]
    else:
        train_vecs = [tuple(rng.randint(0, 3, n).tolist()) for _ in range(n_train)]
    
    # Compute features and labels
    X = []
    y = []
    for vec in train_vecs:
        features = _compute_conflict_features(vec, cap_set)
        X.append(features)
        # Label: 1 if adding vec to cap_set creates a 3-term AP
        test_set = cap_set + [vec]
        label = 1.0 if not _is_cap_set(test_set) else 0.0
        y.append(label)
    
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    # Normalize features
    if X.shape[0] > 0:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-7
        X = (X - mean) / std
    
    # Train MLP
    W1, b1, W2, b2 = _train_mlp(X, y, hidden_size=32, epochs=300, lr=0.005, seed=42)
    
    return (W1, b1, W2, b2, mean, std, cap_set)

def priority(vector, n):
    """
    Priority function using a learned neural surrogate to predict marginal conflict cost.
    """
    v = np.array(vector, dtype=np.int8)
    
    # Get trained model
    model = _get_trained_model(n)
    
    if model is None:
        # Fallback: simple heuristic
        c0 = np.sum(v == 0)
        c1 = np.sum(v == 1)
        c2 = np.sum(v == 2)
        target = n / 3.0
        balance = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
        return -float(balance)
    
    W1, b1, W2, b2, mean, std, cap_set = model
    
    # Compute features
    features = _compute_conflict_features(tuple(v.tolist()), cap_set)
    
    # Normalize
    features = (features - mean) / std
    
    # Predict conflict probability
    conflict_prob = _mlp_predict(features.reshape(1, -1), W1, b1, W2, b2)[0, 0]
    
    # Priority: lower conflict probability is better
    # Add a small bonus for vectors that are close to the cap set (likely safe)
    min_dist = float('inf')
    for cv in cap_set:
        cv_arr = np.array(cv, dtype=np.int8)
        dist = np.sum(v != cv_arr)
        if dist < min_dist:
            min_dist = dist
    dist_bonus = 0.1 / (1.0 + min_dist / n)
    
    return float(-conflict_prob + dist_bonus)
# EVOLVE-BLOCK-END
