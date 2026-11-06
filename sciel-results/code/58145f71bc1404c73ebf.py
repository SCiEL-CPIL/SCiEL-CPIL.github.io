"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 50
    
    # Initialize with Thue-Morse sequence
    signs = np.array([(-1) ** (bin(i).count('1')) for i in range(n)], dtype=np.int8)
    
    # Precompute all arithmetic progressions
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            max_k = (n - a) // d + 1
            if max_k >= 1:
                a0 = a - 1
                idx = np.arange(0, max_k) * d + a0
                progressions.append((a0, d, max_k, idx))
    
    # Build index: for each position, which progressions include it
    pos_to_progs = [[] for _ in range(n)]
    for pi, (a0, d, max_k, idx) in enumerate(progressions):
        for pos in idx:
            pos_to_progs[pos].append(pi)
    
    def compute_all_discs(signs):
        discs = np.zeros(len(progressions), dtype=np.int32)
        for pi, (a0, d, max_k, idx) in enumerate(progressions):
            partial = np.cumsum(signs[idx])
            discs[pi] = np.max(np.abs(partial))
        return discs
    
    current_discs = compute_all_discs(signs)
    current_max = int(np.max(current_discs))
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(123)
    
    def recompute_affected(pos, signs, discs):
        """Recompute discs for progressions affected by flipping pos"""
        for pi in pos_to_progs[pos]:
            a0, d, max_k, idx = progressions[pi]
            partial = np.cumsum(signs[idx])
            discs[pi] = np.max(np.abs(partial))
    
    # Local search
    stall_count = 0
    while time.time() - start_time < time_limit:
        # Find worst progression
        worst_pi = int(np.argmax(current_discs))
        worst_disc = int(current_discs[worst_pi])
        
        if worst_disc <= 1:
            break  # Can't do better than 1
        
        a0, d, max_k, idx = progressions[worst_pi]
        
        # Try flipping each position in the worst progression
        improved = False
        for pos in idx:
            signs[pos] = -signs[pos]
            
            # Recompute affected discs
            old_discs_affected = {}
            for pi in pos_to_progs[pos]:
                old_discs_affected[pi] = current_discs[pi]
                a0_p, d_p, max_k_p, idx_p = progressions[pi]
                partial = np.cumsum(signs[idx_p])
                current_discs[pi] = np.max(np.abs(partial))
            
            new_max = int(np.max(current_discs))
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                improved = True
                break
            else:
                # Revert
                signs[pos] = -signs[pos]
                for pi, old_val in old_discs_affected.items():
                    current_discs[pi] = old_val
        
        if not improved:
            # Random perturbation to escape local minimum
            stall_count += 1
            for _ in range(10):
                pos = rng.randint(0, n - 1)
                signs[pos] = -signs[pos]
                
                old_discs_affected = {}
                for pi in pos_to_progs[pos]:
                    old_discs_affected[pi] = current_discs[pi]
                    a0_p, d_p, max_k_p, idx_p = progressions[pi]
                    partial = np.cumsum(signs[idx_p])
                    current_discs[pi] = np.max(np.abs(partial))
                
                new_max = int(np.max(current_discs))
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                        stall_count = 0
                    break
                else:
                    signs[pos] = -signs[pos]
                    for pi, old_val in old_discs_affected.items():
                        current_discs[pi] = old_val
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
