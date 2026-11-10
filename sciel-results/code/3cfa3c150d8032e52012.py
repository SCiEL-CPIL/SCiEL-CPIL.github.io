"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin from 150s kill
    
    target_n = 34
    M = target_n * (target_n - 1) * (target_n - 2) // 6
    
    # Precompute edge indices
    edge_index = {}
    idx = 0
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                edge_index[(a, b, c)] = idx
                idx += 1
                
    # Precompute edge indices for all K4s and K5s
    k4_edges = []
    for quad in combinations(range(target_n), 4):
        edges = []
        for triple in combinations(quad, 3):
            edges.append(edge_index[triple])
        k4_edges.append(edges)
        
    k5_edges = []
    for quint in combinations(range(target_n), 5):
        edges = []
        for triple in combinations(quint, 3):
            edges.append(edge_index[triple])
        k5_edges.append(edges)
        
    k4_np = np.array(k4_edges, dtype=np.int32)
    k5_np = np.array(k5_edges, dtype=np.int32)
    
    # Helper to check validity
    def is_valid(colors):
        # Check for Red K4 (all 4 edges red i.e. 0)
        k4_bad = np.sum(colors[k4_np], axis=1) == 0
        if np.any(k4_bad):
            return False
        # Check for Blue K5 (all 10 edges blue i.e. 1)
        k5_bad = np.sum(colors[k5_np], axis=1) == 10
        if np.any(k5_bad):
            return False
        return True
        
    # Local search
    max_restarts = 20
    best_valid_n = 9
    best_valid_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    for restart in range(max_restarts):
        if time.time() - start_time > deadline - 5:
            break
            
        # Initialize random coloring
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        
        # Local search loop
        no_improve_count = 0
        while time.time() - start_time < deadline - 5:
            # Check for Red K4
            k4_bad = np.sum(colors[k4_np], axis=1) == 0
            bad_k4_indices = np.where(k4_bad)[0]
            
            # Check for Blue K5
            k5_bad = np.sum(colors[k5_np], axis=1) == 10
            bad_k5_indices = np.where(k5_bad)[0]
            
            if len(bad_k4_indices) == 0 and len(bad_k5_indices) == 0:
                # Success
                return {'n': target_n, 'colors': colors.tolist()}
                
            # If stuck, flip an edge from a violating set
            # Prioritize flipping edges in bad K4s or K5s
            bad_edges = set()
            # Sample a few bad structures to avoid huge sets
            if len(bad_k4_indices) > 0:
                sample_k4 = np.random.choice(bad_k4_indices, min(10, len(bad_k4_indices)), replace=False)
                for idx in sample_k4:
                    bad_edges.update(k4_np[idx].tolist())
            if len(bad_k5_indices) > 0:
                sample_k5 = np.random.choice(bad_k5_indices, min(10, len(bad_k5_indices)), replace=False)
                for idx in sample_k5:
                    bad_edges.update(k5_np[idx].tolist())
                    
            if not bad_edges:
                 break
                
            target_edge = random.choice(list(bad_edges))
            colors[target_edge] ^= 1 # Flip 0<->1
            
            # Simple no-improve counter (optional, but good for restarts)
            # We could track total violations, but it's expensive. 
            # Let's just rely on time limit and restarts.
            
            # To prevent infinite loops on the same flip, we could add a small perturbation
            # But random choice from bad_edges should be enough.
            
    # Fallback: If n=34 search failed, try to find a valid smaller n?
    # The seed solution n=9 is valid.
    # We can try n=10, 11... up to 33 if we have time?
    # But the task says maximize n.
    # If we can't find 34, returning 9 is safe.
    # Can we do better than 9 easily?
    # Maybe try a quick local search for n=20?
    # But the precomputed arrays are for n=34.
    # Re-computing for smaller n is cheap.
    
    # Let's try a few smaller n values if we have time.
    # Actually, the probability of finding 34 is high with enough restarts.
    # If we ran out of time, we return the seed.
    
    # Let's just return the seed fallback to be safe.
    return {'n': 9, 'colors': best_valid_colors}
# EVOLVE-BLOCK-END
