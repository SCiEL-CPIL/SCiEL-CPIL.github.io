import time
import numpy as np

N = 17
TOTAL_MASKS = 1 << N  # 131072

# EVOLVE-BLOCK-START
def _run_network(net, masks):
    """Run network `net` (list of (i,j)) on array of masks (shape (M,)).
    Returns array of sorted masks (shape (M,))."""
    x = masks.astype(np.int32).copy()
    for i, j in net:
        a = (x >> i) & 1
        b = (x >> j) & 1
        # if a > b, swap: set i to 0, j to 1
        swap = (a & ~b)  # a=1, b=0
        x = (x & ~(swap << i)) | (swap << j)
    return x

def _verify(net, masks):
    """Check if net sorts all masks. Returns True if sorted."""
    sorted_masks = _run_network(net, masks)
    return np.all(sorted_masks == masks)

def _count_unsorted(net, masks):
    """Count how many masks are not sorted by net."""
    sorted_masks = _run_network(net, masks)
    return int(np.sum(sorted_masks != masks))

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    masks = np.arange(TOTAL_MASKS, dtype=np.int32)
    
    # Start with seed
    best_net = _build_seed()
    best_count_unsorted = _count_unsorted(best_net, masks)
    if best_count_unsorted == 0:
        best_valid = best_net[:]
    else:
        best_valid = None
    
    best_score = len(best_net) if best_valid else 999999
    
    # Precompute all possible comparator positions
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Local search: try to remove comparators
    if best_valid:
        net = best_valid[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each comparator
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                trial = net[:idx] + net[idx+1:]
                cnt = _count_unsorted(trial, masks)
                if cnt == 0:
                    net = trial
                    improved = True
                    if len(net) < best_score:
                        best_score = len(net)
                        best_valid = net[:]
                    break  # restart removal scan
        best_net = net[:]
    
    # Local search: try to add/replace comparators to reduce unsorted count
    if best_valid is None or _count_unsorted(best_net, masks) > 0:
        # We need to find a valid network. Start from seed and do insertion search.
        net = _build_seed()
        cnt = _count_unsorted(net, masks)
        
        # If seed is valid, we already have it. Otherwise try to fix.
        # For 17 wires, odd-even transposition should be valid. Let's verify.
        if cnt == 0:
            best_valid = net[:]
            best_score = len(net)
    
    # Now try to reduce size further via insertion of useful comparators
    # and removal of redundant ones, in an alternating fashion.
    if best_valid:
        net = best_valid[:]
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Phase 1: try removing comparators
            removed = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                trial = net[:idx] + net[idx+1:]
                cnt = _count_unsorted(trial, masks)
                if cnt == 0:
                    net = trial
                    removed = True
                    if len(net) < best_score:
                        best_score = len(net)
                        best_valid = net[:]
                    break
            
            # Phase 2: try adding a comparator that reduces unsorted count
            if not removed:
                # Find the comparator that most reduces unsorted count
                best_add_idx = -1
                best_add_cnt = _count_unsorted(net, masks)
                # Sample a subset of positions for speed
                import random
                random.seed(42)
                sample = random.sample(all_comps, min(200, len(all_comps)))
                for pos in sample:
                    if time.time() > deadline - 5:
                        break
                    # Try inserting at each position
                    for ins_idx in range(len(net) + 1):
                        trial = net[:ins_idx] + [pos] + net[ins_idx:]
                        cnt = _count_unsorted(trial, masks)
                        if cnt < best_add_cnt:
                            best_add_cnt = cnt
                            best_add_idx = ins_idx
                            best_add_pos = pos
                if best_add_idx >= 0:
                    net = net[:best_add_idx] + [best_add_pos] + net[best_add_idx:]
                    if _count_unsorted(net, masks) == 0 and len(net) < best_score:
                        best_score = len(net)
                        best_valid = net[:]
    
    if best_valid is not None:
        return best_valid
    else:
        # Fallback: return seed (should be valid)
        return _build_seed()
# EVOLVE-BLOCK-END
