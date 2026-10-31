import numpy as np
import itertools
import random
import math

def construct():
    N = 16
    
    # 1. Generate Initial Configuration
    # Use a triangular lattice subset as a robust starting point
    # This ensures d_min is not tiny initially, providing a stable baseline.
    
    # Basis vectors for triangular lattice
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate candidate points in a hexagonal region
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            candidates.append(p)
            
    candidates = np.array(candidates)
    dists_to_origin = np.sum(candidates**2, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    
    # Select 16 points from the densest central region
    # The 19 closest points form a hexagon. We take 16 of them.
    # To be symmetric, we take the 7 innermost (dist 0, 1) and 9 of the 12 outer (dist 3).
    # For simplicity and robustness, let's just take the first 16 in sorted order.
    # This might be slightly asymmetric but provides a valid, dense start.
    init_indices = sorted_indices[:16]
    P = candidates[init_indices].copy()
    
    # Center the configuration
    P -= np.mean(P, axis=0)
    
    # Randomize slightly to break symmetry and avoid exact lattice artifacts
    P += 0.01 * np.random.randn(N, 2)
    
    def calculate_score(P):
        """
        Calculate the squared ratio of max to min pairwise distance.
        Returns inf if any distance is zero (collision).
        """
        # Vectorized distance calculation
        # P is (N, 2)
        # Diff is (N, N, 2)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Get upper triangle indices to avoid duplicates and zero diagonal
        iu = np.triu_indices(N, k=1)
        d_sq = dists_sq[iu]
        
        if len(d_sq) == 0:
            return float('inf')
            
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-10:
            return float('inf')
            
        return (d_max_sq / d_min_sq)
    
    # 2. Simulated Annealing
    # We minimize the score directly.
    # Score is scale-invariant, so we don't need to constrain d_min = 1 explicitly.
    # However, if the scale changes, the score stays the same. 
    # To prevent drift to infinity or zero (which doesn't change score but might cause numerical issues),
    # we can optionally re-normalize, but it's not strictly necessary for the score.
    # Actually, re-normalizing to d_min = 1 helps keep numbers in a reasonable range.
    
    def normalize(P):
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, k=1)
        d_sq = dists_sq[iu]
        d_min = np.sqrt(np.min(d_sq))
        if d_min > 1e-10:
            return P / d_min
        return P

    current_P = normalize(P)
    current_score = calculate_score(current_P)
    
    best_P = current_P.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.1
    T_end = 1e-6
    iterations = 50000
    cooling_rate = (T_end / T_start) ** (1.0 / iterations)
    
    np.random.seed(42)
    
    for it in range(iterations):
        # Perturb a random point
        idx = np.random.randint(0, N)
        step_size = 0.05
        delta = np.random.randn(2) * step_size
        
        new_P = current_P.copy()
        new_P[idx] += delta
        
        # Normalize to keep scale consistent (d_min = 1)
        new_P = normalize(new_P)
        
        new_score = calculate_score(new_P)
        
        # Acceptance probability
        if new_score < current_score:
            current_P = new_P
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_P = current_P.copy()
        else:
            # Delta E is positive (score increased)
            delta_E = new_score - current_score
            # Avoid log(0) or negative temp
            if T_start > 0:
                prob = math.exp(-delta_E / T_start)
                if np.random.rand() < prob:
                    current_P = new_P
                    current_score = new_score
        
        # Cool down
        T_start *= cooling_rate
        
        # Optional: Re-center to prevent drift
        if it % 100 == 0:
            current_P -= np.mean(current_P, axis=0)
            best_P -= np.mean(best_P, axis=0) # Keep best consistent? No, best is a snapshot.
            # Actually, centering doesn't change relative distances, so score is invariant.
            # We just need to ensure the returned array is centered for cleanliness.
            
    return best_P
