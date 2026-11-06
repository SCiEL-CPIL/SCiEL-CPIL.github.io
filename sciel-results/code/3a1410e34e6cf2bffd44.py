import time
import itertools
import random

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
TRIPLE_PAIRS = [set(itertools.combinations(t, 2)) for t in ALL_TRIPLES]
TRIPLE_MASKS = [sum(1 << x for x in t) for t in ALL_TRIPLES]
UNIVERSE_MASK = (1 << N) - 1

# Precompute popcount for all 16-bit masks
POP = [0] * (1 << N)
for i in range(1, 1 << N):
    POP[i] = POP[i >> 1] + (i & 1)

def calc_alpha_from_masks(masks):
    """
    Computes the independence number (max size of a set containing no triple).
    A set S is independent if it contains no triple from the system.
    We mark all supersets of each triple mask as 'dependent' (not independent).
    """
    if not masks:
        return N
        
    # independent[i] is True if subset i is independent
    independent = [True] * (1 << N)
    
    for m in masks:
        # Iterate all supersets of m within the universe
        # Supersets of m are m | z where z is a subset of (~m & UNIVERSE_MASK)
        y = (~m) & UNIVERSE_MASK
        z = y
        while True:
            independent[m | z] = False
            if z == 0:
                break
            z = (z - 1) & y
        
        # Early exit check: if we've already marked everything, alpha is 0? 
        # No, we need the max popcount of remaining independent sets.
        # But we can't easily exit early without checking all.
        # However, if the number of remaining independent sets is small, we can speed up the count.
        
    max_alpha = 0
    for i in range(1 << N):
        if independent[i]:
            c = POP[i]
            if c > max_alpha:
                max_alpha = c
    return max_alpha

def greedy_backtracking(seed_val, max_time=0.5):
    """
    Attempts to build a linear triple system using randomized backtracking.
    Returns (alpha, triples).
    """
    random.seed(seed_val)
    start = time.time()
    
    # We want to maximize the number of triples to minimize alpha.
    # Greedy approach with backtracking limited depth.
    
    # Shuffle all triples
    indices = list(range(len(ALL_TRIPLES)))
    random.shuffle(indices)
    
    triples = []
    used_pairs = set()
    masks = []
    
    # Simple greedy first
    for idx in indices:
        t = ALL_TRIPLES[idx]
        pairs = TRIPLE_PAIRS[idx]
        if pairs.isdisjoint(used_pairs):
            triples.append(t)
            used_pairs |= pairs
            masks.append(TRIPLE_MASKS[idx])
            
    # Try to improve with swaps? Or just restart with different shuffles?
    # The current seed program does random restarts of greedy.
    # Let's try a slightly smarter greedy: prioritize triples that "cover" more unused points?
    # Or just run many greedy restarts.
    
    alpha = calc_alpha_from_masks(masks)
    return alpha, triples

def local_search(triples, seed_val, max_time=1.0):
    """
    Improves a given set of triples by swapping or adding/removing.
    """
    random.seed(seed_val)
    start = time.time()
    
    # Convert to list of tuples for easy manipulation
    current_triples = list(triples)
    used_pairs = set()
    for t in current_triples:
        used_pairs |= TRIPLE_PAIRS[ALL_TRIPLES.index(t)] # This is slow, need index mapping
        # Let's precompute index mapping
        # Actually, let's keep masks and pairs separate.
        
    # This implementation is getting complicated with indices.
    # Let's stick to the fast greedy restarts which seem to be the main driver in the seed.
    # The seed score 1.5714 corresponds to alpha=10 (16/10 = 1.6? No, 1.5714 = 11/7? 16/10.18?
    # Score = seed/value. If seed=1.0, value is the score.
    # If score is 1.5714, and it's seed/value, then value = 1/1.5714 = 0.636? No.
    # "The search score is seed/value, so 1.0 equals the seed program".
    # So if seed program has value V_seed, score_seed = 1.0 => seed = V_seed.
    # Current score 1.5714 means Value = Seed / 1.5714.
    # Wait, "1.0 equals the seed program". So Seed Program Score = 1.0.
    # If Score = Seed / Value, then 1.0 = Seed / Value_seed => Seed = Value_seed.
    # So Value is the independence number?
    # "The score is the largest point set containing no triple... Minimize."
    # So Value = Alpha.
    # If Score = Seed / Alpha, and Score_seed = 1.0, then Seed = Alpha_seed.
    # Current Score 1.5714 => 1.5714 = Seed / Alpha_current.
    # If Seed is fixed (e.g., 11?), then Alpha_current = 11 / 1.5714 = 7.
    # Or if Seed = Alpha_seed = 11, then 1.5714 = 11 / Alpha_current => Alpha_current = 7.
    # So we are currently getting alpha = 7.
    # We want to get alpha = 6 or 5.
    
    # Let's verify: 11/7 = 1.5714. Yes.
    # So we need to reduce alpha from 7 to 6.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_alpha = N + 1
    best_triples = []
    
    num_restarts = 0
    while time.time() < deadline:
        seed = num_restarts
        alpha, triples = greedy_backtracking(seed)
        
        # If we found a better alpha, update
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 5: # Target
                break
        
        num_restarts += 1
        
    return best_triples
