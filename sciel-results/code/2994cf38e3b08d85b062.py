import time
import random

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute all triples (masks)
TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))

NUM_TRIPLES = len(TRIPLES)

# Precompute supersets for each triple
# supersets[i] is a list of masks s such that TRIPLES[i] is a subset of s
SUPERSETS = [None] * NUM_TRIPLES
for i, t in enumerate(TRIPLES):
    comp = ((1 << N_POINTS) - 1) ^ t
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

# Map mask to index for fast lookup
MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}

# Sort masks by popcount descending for fast alpha calculation
# We only care about masks with popcount > current best alpha potentially
# But for simplicity, sort all
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)

# Precompute list of triples for each mask? No, too big.
# We will use the "blocked_count" approach.

def construct():
    deadline = time.time() + 110.0
    _random = random.Random(42)
    
    # State
    # triples_set: set of triple masks currently in the system
    # blocked_count: array of size 2^16, blocked_count[m] = number of triples in system subset of m
    # is_in_system: boolean array or set for fast membership check
    
    triples_set = set()
    blocked_count = [0] * N_MASKS
    is_in_system = [False] * NUM_TRIPLES
    
    def add_triple(idx):
        """Add triple at index idx to the system."""
        t_mask = TRIPLES[idx]
        if is_in_system[idx]:
            return
        is_in_system[idx] = True
        triples_set.add(t_mask)
        # Update blocked counts
        for s in SUPERSETS[idx]:
            blocked_count[s] += 1
            
    def remove_triple(idx):
        """Remove triple at index idx from the system."""
        t_mask = TRIPLES[idx]
        if not is_in_system[idx]:
            return
        is_in_system[idx] = False
        triples_set.remove(t_mask)
        # Update blocked counts
        for s in SUPERSETS[idx]:
            blocked_count[s] -= 1

    def get_alpha_and_set():
        """Find the maximum independent set size and one such set."""
        # An independent set is a mask with blocked_count[mask] == 0
        # We iterate from largest popcount down
        for s in SORTED_MASKS:
            if blocked_count[s] == 0:
                return POP_COUNT[s], s
        return 0, 0

    def can_add_mask(t_mask):
        """Check if a triple mask can be added to the current system (linearity check)."""
        # Check against all triples in system
        # Optimization: Iterate over system triples
        for t in triples_set:
            if (t & t_mask).bit_count() >= 2:
                return False
        return True

    # 1. Greedy Initialization
    # Randomly shuffle triple indices
    indices = list(range(NUM_TRIPLES))
    _random.shuffle(indices)
    
    for idx in indices:
        t_mask = TRIPLES[idx]
        if can_add_mask(t_mask):
            add_triple(idx)
            
    current_alpha, current_set = get_alpha_and_set()
    best_alpha = current_alpha
    best_system = triples_set.copy()
    best_blocked = blocked_count.copy() # Maybe not needed if we reconstruct from best_system at the end
    
    # We need to track the actual triples for the best solution
    # Since we mutate the state, we should save the list of indices for the best solution
    best_indices = [i for i in range(NUM_TRIPLES) if is_in_system[i]]

    # 2. Directed Local Search
    while time.time() < deadline:
        # Update current alpha
        current_alpha, current_set = get_alpha_and_set()
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_indices = [i for i in range(NUM_TRIPLES) if is_in_system[i]]
            
        # If alpha is 0 or 1, we are optimal (theoretically min is >1 for N=16? 
        # Actually, if we cover all points, alpha is small. 
        # Min possible alpha for N=16? 
        # If we have a Steiner system, alpha is around 6-7.
        # Let's keep searching until deadline.
        
        # Find candidates: Triples T such that T is subset of current_set
        # Since current_set is independent, no triple in system is subset of it.
        # So all candidates are NOT in the system.
        candidates = []
        # Iterate all triples, check if subset of current_set
        # Optimization: Only check triples with size 3 (all of them)
        # t_mask & ~current_set == 0
        not_in_set_mask = ((1 << N_POINTS) - 1) ^ current_set
        
        for i in range(NUM_TRIPLES):
            if is_in_system[i]:
                continue
            t_mask = TRIPLES[i]
            if (t_mask & not_in_set_mask) == 0:
                candidates.append(i)
                
        if not candidates:
            # No triple is fully contained in the current max independent set.
            # This means the current max independent set is "hard" to break directly.
            # We might need to change the structure more significantly.
            # Fallback: Random swap
            if not triples_set:
                continue
                
            # Pick a random triple to remove
            idx_out = _random.choice(list(triples_set))
            # Wait, triples_set contains masks. We need index.
            # Let's maintain a list of active indices for faster selection
            # Or just pick a random index from 0..559 and check if active
            
            active_indices = [i for i in range(NUM_TRIPLES) if is_in_system[i]]
            if not active_indices:
                continue
            idx_out = _random.choice(active_indices)
            
            # Pick a random triple to add
            idx_in = _random.randrange(NUM_TRIPLES)
            if is_in_system[idx_in]:
                continue
                
            # Check linearity: Remove idx_out, then check if idx_in can be added
            remove_triple(idx_out)
            if can_add_mask(TRIPLES[idx_in]):
                add_triple(idx_in)
                # Acceptance: always accept swap in this fallback? 
                # Or check alpha?
                # Let's just accept to explore.
            else:
                # Revert
                add_triple(idx_out)
            continue
            
        # We have candidates. Pick one at random.
        idx_in = _random.choice(candidates)
        t_in_mask = TRIPLES[idx_in]
        
        # Check if it can be added directly
        if can_add_mask(t_in_mask):
            add_triple(idx_in)
            # This should reduce alpha (or keep same if another set exists)
            # We accept this move immediately as it is "directed" towards reducing alpha
            continue
            
        # It cannot be added directly. Find conflicts.
        # Conflicts are triples in system that share >= 2 points with t_in_mask
        conflicts = []
        for t in triples_set:
            if (t & t_in_mask).bit_count() >= 2:
                # Find index of t
                # We can use MASK_TO_INDEX
                idx_conf = MASK_TO_INDEX[t]
                conflicts.append(idx_conf)
                
        if not conflicts:
            # This shouldn't happen if can_add_mask returned False
            # Unless can_add_mask has a bug or state is inconsistent
            continue
            
        # Try to swap: Remove one conflict, add candidate
        idx_out = _random.choice(conflicts)
        
        # Perform swap
        remove_triple(idx_out)
        
        # Check if we can add idx_in now
        if can_add_mask(t_in_mask):
            add_triple(idx_in)
            # Check if this improved or maintained alpha
            # We don't calculate alpha immediately to save time, 
            # but we know we broke the current max set.
            # We will calculate alpha in the next loop iteration.
            
            # Heuristic: If we just did a swap, we might be stuck in a local min.
            # If alpha doesn't improve after a few swaps, we might need to restart or randomize more.
            # For now, accept the swap.
        else:
            # Revert swap
            add_triple(idx_out)
            
    # Convert best_indices to result format
    result = []
    for idx in best_indices:
        mask = TRIPLES[idx]
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
