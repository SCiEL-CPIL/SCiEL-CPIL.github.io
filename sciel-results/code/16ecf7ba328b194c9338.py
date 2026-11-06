# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq

N = 256
DEADLINE = 55.0  # safety margin

# Global precomputed data
_APS = []
_AP_INDICES = [] # List of numpy arrays of indices for each AP
_APS_CONTAINING = [[] for _ in range(N)] # List of AP indices for each position

def _precompute():
    global _APS, _AP_INDICES, _APS_CONTAINING
    _APS = []
    _AP_INDICES = []
    _APS_CONTAINING = [[] for _ in range(N)]
    
    ap_idx = 0
    for d in range(1, N):
        for a in range(N):
            # Generate AP: a, a+d, a+2d, ... < N
            seq = list(range(a, N, d))
            if len(seq) > 0:
                arr = np.array(seq, dtype=np.int32)
                _AP_INDICES.append(arr)
                # Register which APs contain which indices
                for idx in seq:
                    _APS_CONTAINING[idx].append(ap_idx)
                ap_idx += 1
    # Note: The previous code included d up to N-1.
    # My loop: d in 1..N-1. a in 0..N-1.
    # This matches the "any start, positive step" definition.
    
def _initial_sums(signs):
    """Compute sums for all APs."""
    sums = np.zeros(len(_AP_INDICES), dtype=np.int32)
    for i, ap_idx in enumerate(range(len(_AP_INDICES))):
        sums[ap_idx] = signs[_AP_INDICES[ap_idx]].sum()
    return sums

def _build_heap(sums):
    """Build max-heap based on absolute sums."""
    heap = []
    for i, s in enumerate(sums):
        heapq.heappush(heap, (-abs(s), i))
    return heap

def _get_max_from_heap(heap, sums):
    """Get the current max absolute sum, cleaning stale entries."""
    while heap:
        val, idx = heap[0]
        if abs(sums[idx]) == -val:
            return -val
        heapq.heappop(heap)
    return 0 # Should not happen

def construct():
    global _APS, _AP_INDICES, _APS_CONTAINING
    
    t0 = time.time()
    _precompute()
    
    best_signs = None
    best_val = float('inf')
    
    # Try a few random restarts
    for trial in range(5):
        if time.time() - t0 > DEADLINE - 10:
            break
            
        rng = np.random.RandomState(42 + trial)
        signs = rng.choice([-1, 1], size=N).astype(np.int8)
        
        # Initial evaluation
        sums = _initial_sums(signs)
        heap = _build_heap(sums)
        current_max = _get_max_from_heap(heap, sums)
        
        # Local Search with Incremental Updates
        improved = True
        no_improve_count = 0
        max_no_improve = 5000 # Restart local search if stuck
        
        while improved and (time.time() - t0) < DEADLINE - 2:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random index to flip
            i = rng.randint(N)
            old_sign = signs[i]
            delta = -2 * old_sign
            
            # Update sums for APs containing i
            affected_aps = _APS_CONTAINING[i]
            for ap_idx in affected_aps:
                sums[ap_idx] += delta
                heapq.heappush(heap, (-abs(sums[ap_idx]), ap_idx))
            
            # Check new max
            new_max = _get_max_from_heap(heap, sums)
            
            if new_max < current_max:
                # Accept
                signs[i] = -old_sign
                current_max = new_max
                best_val_local = current_max
                if best_val_local < best_val:
                    best_val = best_val_local
                    best_signs = signs.copy()
                improved = True
                no_improve_count = 0
            else:
                # Revert
                signs[i] = old_sign
                for ap_idx in affected_aps:
                    sums[ap_idx] -= delta
                    heapq.heappush(heap, (-abs(sums[ap_idx]), ap_idx))
                
                # Optional: Clean up heap if it gets too large
                if len(heap) > 100000:
                    heap = _build_heap(sums)

    # Final Fallback if no improvement found or time out
    if best_signs is None:
        # Return last trial's signs or random
        if 'signs' in locals():
            best_signs = signs
        else:
            best_signs = np.random.RandomState(0).choice([-1, 1], size=N).astype(np.int8)
            
    # Final verification (optional, but good for safety)
    # The evaluator will compute the score, so we just need to return the signs.
    # We trust our local search tracking.
    
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
