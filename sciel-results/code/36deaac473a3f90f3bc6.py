import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30, seed=0):
    """Run ACO with given heuristic, return best total prize."""
    rng = np.random.RandomState(seed)
    n = len(prizes)
    d = distance_matrix
    d_to_depot = d[:, 0]
    
    best_prize = 0.0
    pheromone = np.ones((n, n))
    # Precompute eligibility: which nodes j can be reached from i
    # Ant at i with length_so_far can go to j if:
    #   length_so_far + d[i,j] + d[j,0] <= max_len
    # and j not visited, j != 0
    
    for it in range(n_iters):
        iter_best = 0.0
        for ant in range(n_ants):
            visited = [0]
            length_so_far = 0.0
            total_prize = 0.0
            current = 0
            
            while True:
                # Find eligible next nodes
                eligible = []
                for j in range(1, n):
                    if j in visited:
                        continue
                    if length_so_far + d[current, j] + d[j, 0] <= max_len:
                        eligible.append(j)
                
                if not eligible:
                    break
                
                # Compute probabilities: pheromone[i,j] * heuristic[i,j]
                probs = np.array([pheromone[current, j] * heuristic[current, j] for j in eligible])
                probs = np.maximum(probs, 1e-9)
                total_prob = probs.sum()
                if total_prob <= 0:
                    break
                probs = probs / total_prob
                
                # Roulette wheel selection
                r = rng.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(eligible):
                    idx = len(eligible) - 1
                
                next_node = eligible[idx]
                length_so_far += d[current, next_node]
                total_prize += prizes[next_node]
                visited.append(next_node)
                current = next_node
            
        # End of ant
        if total_prize > iter_best:
            iter_best = total_prize
        if total_prize > best_prize:
            best_prize = total_prize
    
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
    
    return best_prize


def _baseline_heuristic(prizes, distance_matrix, max_len):
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    d_to_depot = distance_matrix[:, 0]
    num = prizes[None, :] + 1e-9
    denom = distance_matrix + d_to_depot[None, :] + 1e-9
    eta = num / denom
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    return eta


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin
    
    # Start with baseline heuristic
    best_heuristic = _baseline_heuristic(prizes, distance_matrix, max_len)
    best_score = _run_aco(prizes, distance_matrix, max_len, best_heuristic, n_ants=20, n_iters=30, seed=42)
    
    # Try various heuristic parameterizations and pick the best
    # Heuristic form: prize[j]^alpha / (d[i,j] + d[j,0])^beta
    # We'll search over alpha, beta combinations
    
    candidates = []
    d_to_depot = distance_matrix[:, 0]
    
    # Generate candidate heuristic matrices
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            num = (prizes[None, :] + 1e-9) ** alpha
            denom = (distance_matrix + d_to_depot[None, :] + 1e-9) ** beta
            eta = num / denom
            eta[~np.isfinite(eta)] = 1e-9
            eta[eta <= 0] = 1e-9
            candidates.append(eta)
    
    # Also try prize-weighted distance heuristic
    for w in [0.5, 1.0, 2.0]:
        num = prizes[None, :] + 1e-9
        denom = distance_matrix + d_to_depot[None, :] + 1e-9
        eta = (num ** w) / denom
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        candidates.append(eta)
    
    # Try reciprocal distance heuristic with prize scaling
    for s in [1.0, 2.0, 5.0]:
        eta = (s * prizes[None, :] + 1e-9) / (distance_matrix + d_to_depot[None, :] + 1e-9)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        candidates.append(eta)
    
    # Evaluate candidates with quick ACO
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        score = _run_aco(prizes, distance_matrix, max_len, cand, n_ants=20, n_iters=30, seed=42)
        if score > best_score:
            best_score = score
            best_heuristic = cand
    
    # If we have time, try more refined search around the best
    if time.time() < deadline - 20:
        # Try perturbed versions of the best heuristic
        base = best_heuristic.copy()
        for pert in [0.5, 0.8, 1.2, 1.5]:
            if time.time() > deadline - 10:
                break
            perturbed = base * pert
            perturbed[~np.isfinite(perturbed)] = 1e-9
            perturbed[perturbed <= 0] = 1e-9
            score = _run_aco(prizes, distance_matrix, max_len, perturbed, n_ants=20, n_iters=30, seed=42)
            if score > best_score:
                best_score = score
                best_heuristic = perturbed
    
    # Final sanitization
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    best_heuristic[best_heuristic <= 0] = 1e-9
    
    return best_heuristic
