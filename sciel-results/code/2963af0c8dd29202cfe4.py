import numpy as np
import time
from collections import defaultdict

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=42, return_paths=False):
    """
    Simulates ACO. If return_paths is True, returns (best_prize, list_of_paths).
    Otherwise returns best_prize.
    """
    n = len(prizes)
    H = np.where(heuristic > 0, heuristic, 1e-9)
    
    best_prize = 0
    best_paths = []
    rng = np.random.default_rng(seed)
    
    for it in range(num_iters):
        best_iter_prize = 0
        iter_paths = []
        
        for _ in range(num_ants):
            current_node = 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True
            path_len = 0
            total_prize = prizes[0]
            path = [0]
            
            while True:
                remaining_budget = max_len - path_len
                valid_indices = np.where(
                    (~visited_mask) & 
                    (distance_matrix[current_node, :] + distance_matrix[:, 0] <= remaining_budget)
                )[0]
                
                if len(valid_indices) == 0:
                    break
                
                probs = H[current_node, valid_indices]
                total_prob = np.sum(probs)
                if total_prob == 0:
                    break
                
                probs = probs / total_prob
                next_idx_in_valid = rng.choice(len(valid_indices), p=probs)
                next_node = valid_indices[next_idx_in_valid]
                
                path_len += distance_matrix[current_node, next_node]
                total_prize += prizes[next_node]
                visited_mask[next_node] = True
                current_node = next_node
                path.append(next_node)
            
        if total_prize > best_iter_prize:
            best_iter_prize = total_prize
            if return_paths:
                iter_paths.append((total_prize, path))
        
        if total_prize > best_prize:
            best_prize = total_prize
            if return_paths:
                best_paths.append((total_prize, path))
                
    if return_paths:
        return best_prize, best_paths
    return best_prize

def _generate_training_data(prizes, distance_matrix, max_len, heuristic, num_ants=50, num_iters=5, seed=123):
    """
    Generates positive and negative examples for logistic regression.
    Positive: Moves made by the top K% of ants (or just the best ant).
    Negative: Other valid moves at the same state.
    """
    n = len(prizes)
    
    # Run a diverse simulation to collect good paths
    # We use a neutral heuristic first to get a baseline, or the current best if available
    # Here we assume we are learning from scratch or refining. 
    # Let's run the simulation with the current heuristic to see what works.
    
    best_prize, all_paths = _simulate_aco(
        prizes, distance_matrix, max_len, heuristic, 
        num_ants=num_ants, num_iters=num_iters, seed=seed, return_paths=True
    )
    
    if not all_paths:
        return None, None
        
    # Sort paths by prize
    all_paths.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20% of paths as "expert" demonstrations
    k = max(1, len(all_paths) // 5)
    expert_paths = [p for _, p in all_paths[:k]]
    
    # Collect features
    # State: (i, j) where i is current, j is next
    # We want to learn H[i, j] such that high H[i, j] correlates with good moves.
    
    # We will build a dataset of (features, label)
    # Features for edge (i, j):
    # 1. prize[j]
    # 2. dist[i, j]
    # 3. dist[j, 0]
    # 4. prize[j] / (dist[i, j] + dist[j, 0])
    # 5. dist[i, j] / (dist[i, j] + dist[j, 0])
    
    pos_examples = []
    neg_examples = []
    
    # To ensure we have enough negatives, we sample states from expert paths
    # and look at all valid next nodes.
    
    for path in expert_paths:
        for step_idx in range(len(path) - 1):
            i = path[step_idx]
            j_chosen = path[step_idx + 1]
            
            # Determine all valid next nodes from i given the state (visited so far)
            # Note: In ACO, visited set matters. 
            # For simplicity in feature extraction, we approximate the "state" 
            # by just the current node i and the remaining budget implied by the path length so far.
            # However, the validity constraint depends on the specific visited set.
            # To be rigorous, we should reconstruct the visited mask.
            
            visited = np.zeros(n, dtype=bool)
            for node in path[:step_idx+1]:
                visited[node] = True
            
            path_len_so_far = 0
            for k in range(step_idx):
                path_len_so_far += distance_matrix[path[k], path[k+1]]
            
            remaining_budget = max_len - path_len_so_far
            
            valid_indices = np.where(
                (~visited) & 
                (distance_matrix[i, :] + distance_matrix[:, 0] <= remaining_budget)
            )[0]
            
            if len(valid_indices) == 0:
                continue
                
            # Positive example: the chosen j
            pos_examples.append((i, j_chosen))
            
            # Negative examples: other valid nodes
            for j_other in valid_indices:
                if j_other != j_chosen:
                    neg_examples.append((i, j_other))
                    
    # If we don't have enough data, return None
    if len(pos_examples) < 10:
        return None, None
        
    return pos_examples, neg_examples

def _build_features(pos_examples, neg_examples, prizes, distance_matrix):
    """
    Build the feature matrix X and label vector y for logistic regression.
    """
    all_examples = []
    
    for i, j in pos_examples:
        all_examples.append((i, j, 1))
    for i, j in neg_examples:
        all_examples.append((i, j, 0))
        
    # Shuffle to avoid bias in training
    import random
    random.shuffle(all_examples)
    
    X = []
    y = []
    
    for i, j, label in all_examples:
        d_ij = distance_matrix[i, j]
        d_j0 = distance_matrix[j, 0]
        p_j = prizes[j]
        
        # Avoid division by zero
        d_safe = max(d_ij, 1e-9)
        d_ret_safe = max(d_j0, 1e-9)
        cost_safe = max(d_ij + d_j0, 1e-9)
        
        features = [
            p_j,
            d_ij,
            d_j0,
            p_j / cost_safe,       # Efficiency
            p_j / d_safe,          # Local ratio
            1.0 / (1.0 + d_ij),    # Inverse distance (smooth)
            1.0 / (1.0 + d_j0),    # Inverse return distance
            p_j * (1.0 / cost_safe) # Interaction
        ]
        
        X.append(features)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    # Normalize features to prevent numerical issues in logistic regression
    # Simple standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    
    return X_norm, y, mean, std

def _logistic_regression_train(X, y, lr=0.1, epochs=1000, l2_lambda=0.1):
    """
    Simple logistic regression with gradient descent.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    for epoch in range(epochs):
        # Forward pass
        logits = X @ weights + bias
        # Clip to avoid overflow in sigmoid
        logits = np.clip(logits, -500, 500)
        probs = 1 / (1 + np.exp(-logits))
        
        # Gradient
        error = probs - y
        grad_w = (X.T @ error) / n_samples + l2_lambda * weights
        grad_b = np.mean(error)
        
        # Update
        weights -= lr * grad_w
        bias -= lr * grad_b
        
    return weights, bias

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290
    
    n = len(prizes)
    
    # 1. Initial heuristic: simple prize/distance
    d_safe = np.maximum(distance_matrix, 1e-9)
    p_safe = np.maximum(prizes, 1e-9)
    base_h = p_safe[None, :] / d_safe
    base_h = np.where(base_h > 0, base_h, 1e-9)
    
    # 2. Generate training data from a simulation using base_h
    pos_ex, neg_ex = _generate_training_data(prizes, distance_matrix, max_len, base_h)
    
    best_h = base_h.copy()
    best_score = _simulate_aco(prizes, distance_matrix, max_len, best_h)
    
    if pos_ex is not None:
        # 3. Build features and train logistic regression
        X, y, mean, std = _build_features(pos_ex, neg_ex, prizes, distance_matrix)
        
        if X is not None and len(X) > 0:
            weights, bias = _logistic_regression_train(X, y)
            
            # 4. Construct new heuristic from learned weights
            # We need to apply the learned weights to ALL edges (i, j)
            # H[i, j] = sigmoid( sum( w_k * feature_k(i, j) ) + bias )
            # Or simply the linear score, clipped to positive.
            
            H_learned = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    
                    d_ij = distance_matrix[i, j]
                    d_j0 = distance_matrix[j, 0]
                    p_j = prizes[j]
                    
                    d_safe = max(d_ij, 1e-9)
                    d_ret_safe = max(d_j0, 1e-9)
                    cost_safe = max(d_ij + d_j0, 1e-9)
                    
                    features = [
                        p_j,
                        d_ij,
                        d_j0,
                        p_j / cost_safe,
                        p_j / d_safe,
                        1.0 / (1.0 + d_ij),
                        1.0 / (1.0 + d_j0),
                        p_j * (1.0 / cost_safe)
                    ]
                    
                    # Normalize features using the mean/std from training
                    features_norm = (np.array(features) - mean) / std
                    
                    logit = np.dot(weights, features_norm) + bias
                    # Use sigmoid to map to (0, 1)
                    prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
                    
                    # Scale up to make it comparable to base_h magnitudes if needed,
                    # but ACO normalizes probabilities per step, so scale doesn't matter as much,
                    # only relative ranking. However, very small values might underflow.
                    H_learned[i, j] = max(prob, 1e-9)
            
            # Ensure diagonal is 0 or small (self-loops are invalid in path logic usually, 
            # but the mask handles it. Let's set to 0 to be safe, or 1e-9)
            np.fill_diagonal(H_learned, 1e-9)
            
            # 5. Evaluate the learned heuristic
            if time.time() < deadline:
                score_learned = _simulate_aco(prizes, distance_matrix, max_len, H_learned)
                if score_learned > best_score:
                    best_score = score_learned
                    best_h = H_learned
            else:
                # If time is tight, just return what we have
                pass
                
    # 6. Try to refine with a second pass?
    # If we have time, use the best_h found to generate better training data
    if time.time() < deadline - 30:
        pos_ex2, neg_ex2 = _generate_training_data(prizes, distance_matrix, max_len, best_h, seed=456)
        if pos_ex2 is not None:
            X2, y2, mean2, std2 = _build_features(pos_ex2, neg_ex2, prizes, distance_matrix)
            if X2 is not None and len(X2) > 0:
                weights2, bias2 = _logistic_regression_train(X2, y2)
                
                H_refined = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        d_ij = distance_matrix[i, j]
                        d_j0 = distance_matrix[j, 0]
                        p_j = prizes[j]
                        
                        d_safe = max(d_ij, 1e-9)
                        cost_safe = max(d_ij + d_j0, 1e-9)
                        
                        features = [
                            p_j,
                            d_ij,
                            d_j0,
                            p_j / cost_safe,
                            p_j / d_safe,
                            1.0 / (1.0 + d_ij),
                            1.0 / (1.0 + d_j0),
                            p_j * (1.0 / cost_safe)
                        ]
                        features_norm = (np.array(features) - mean2) / std2
                        logit = np.dot(weights2, features_norm) + bias2
                        prob = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
                        H_refined[i, j] = max(prob, 1e-9)
                        
                np.fill_diagonal(H_refined, 1e-9)
                
                if time.time() < deadline:
                    score_refined = _simulate_aco(prizes, distance_matrix, max_len, H_refined)
                    if score_refined > best_score:
                        best_score = score_refined
                        best_h = H_refined

    return best_h
