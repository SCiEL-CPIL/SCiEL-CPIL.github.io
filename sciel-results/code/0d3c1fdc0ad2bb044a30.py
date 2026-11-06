"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
import sys

N = 16
DEADLINE = time.time() + 110  # Safety margin before 120s kill

# Precompute all triples and their pair masks
ALL_TRIPLES = []
TRIPLE_MASKS = {}
for t in itertools.combinations(range(N), 3):
    m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    ALL_TRIPLES.append(t)
    TRIPLE_MASKS[t] = m

# Map triple index to mask for fast access
T_MASKS = [TRIPLE_MASKS[t] for t in ALL_TRIPLES]

# Precompute conflicts: for each triple i, list of j such that i and j share a pair
# Two triples share a pair if they share exactly 2 points.
# We can check this via bitmasks: (m1 & m2).bit_count() == 2
N_TRIPLES = len(ALL_TRIPLES)
CONFLICTS = [[] for _ in range(N_TRIPLES)]
for i in range(N_TRIPLES):
    m1 = T_MASKS[i]
    for j in range(i + 1, N_TRIPLES):
        m2 = T_MASKS[j]
        if (m1 & m2).bit_count() == 2:
            CONFLICTS[i].append(j)
            CONFLICTS[j].append(i)

def triple_pairs(t):
    a, b, c = t
    return ((a, b) if a < b else (b, a),
            (a, c) if a < c else (c, a),
            (b, c) if b < c else (c, b))

def is_linear(triples):
    """Check linearity using pair sets."""
    seen = set()
    for t in triples:
        for p in triple_pairs(t):
            if p in seen:
                return False
            seen.add(p)
    return True

def compute_alpha_fast(triples):
    """
    Compute independence number by precomputing independent masks.
    Returns the size of the largest independent set.
    """
    if not triples:
        return N
    
    # independent[mask] will be True if mask contains no triple from `triples`
    # We start with all True, then mark supersets of each triple as False
    # Optimization: Use a list of booleans or bitsets.
    # Size 2^16 = 65536.
    indep = [True] * (1 << N)
    
    # For each triple in the system, mark all supersets as non-independent
    for t in triples:
        m = TRIPLE_MASKS[t]
        # Iterate over all masks that are supersets of m
        # Supersets of m are m | x where x is a subset of (~m) & ((1<<N)-1)
        # We can iterate submasks of the complement
        comp = ((1 << N) - 1) ^ m
        sub = comp
        while True:
            mask = m | sub
            indep[mask] = False
            if sub == 0:
                break
            sub = (sub - 1) & comp
    
    # Find max popcount among independent masks
    max_alpha = 0
    for mask in range(1 << N):
        if indep[mask]:
            cnt = mask.bit_count()
            if cnt > max_alpha:
                max_alpha = cnt
                # Early exit if we reach N? Unlikely for dense systems.
                if max_alpha == N:
                    break
    return max_alpha

def greedy_construct():
    """
    Greedy construction of a linear triple system.
    Tries to pack as many triples as possible.
    """
    triples = []
    used_pairs = set()
    available = list(range(N_TRIPLES))
    random.shuffle(available)
    
    for idx in available:
        t = ALL_TRIPLES[idx]
        ps = triple_pairs(t)
        if not any(p in used_pairs for p in ps):
            triples.append(t)
            for p in ps:
                used_pairs.add(p)
            
    return triples

def local_search(triples, time_limit=5.0):
    """
    Local search: swap triples to reduce alpha.
    Uses compute_alpha_fast which is reasonably fast.
    """
    if not triples:
        return triples, N
    
    start_time = time.time()
    cur_alpha = compute_alpha_fast(triples)
    triples = list(triples)
    n = len(triples)
    
    # Precompute indices of triples in the list for fast lookup?
    # Or just work with the list.
    
    # To speed up conflict checking during swap:
    # Maintain a set of used pairs? Or just check against other triples.
    # Since n is small (~40), checking against all others is O(n).
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try each position
        for i in range(n):
            if time.time() - start_time > time_limit:
                return triples, cur_alpha
            
            old_t = triples[i]
            old_pairs = set(triple_pairs(old_t))
            
            # Candidates: all triples not in current list and not conflicting with current list (except i)
            # To be efficient, we can just iterate all ALL_TRIPLES and check validity.
            # But that's 560 checks per i, 40 i's -> 22400 checks per epoch.
            # Each check involves O(n) conflict check. 22400 * 40 = ~1M ops. OK.
            
            # Optimization: Precompute which triples are currently in use
            current_set = set(triples)
            
            for new_idx, new_t in enumerate(ALL_TRIPLES):
                if new_t in current_set:
                    continue
                
                new_pairs = set(triple_pairs(new_t))
                
                # Check if new_t conflicts with any other triple in triples (excluding i)
                valid = True
                for j, other_t in enumerate(triples):
                    if j == i:
                        continue
                    # Check pair overlap
                    # Since triples are linear, we can just check if they share a pair
                    # Quick bit check:
                    if (TRIPLE_MASKS[new_t] & TRIPLE_MASKS[other_t]).bit_count() == 2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Swap
                triples[i] = new_t
                current_set.remove(old_t)
                current_set.add(new_t)
                
                new_alpha = compute_alpha_fast(triples)
                
                if new_alpha < cur_alpha:
                    cur_alpha = new_alpha
                    improved = True
                    break # Restart epoch or continue? Continue to find more
                else:
                    # Revert
                    triples[i] = old_t
                    current_set.remove(new_t)
                    current_set.add(old_t)
            
            if improved:
                # After an improvement, we can break inner loop to restart scanning
                # Or just continue. Breaking is safer to avoid missing other spots immediately.
                pass 

    return triples, cur_alpha

def construct():
    best_triples = None
    best_alpha = N
    start_time = time.time()
    
    # 1. Generate several greedy seeds
    seeds = []
    for _ in range(10):
        if time.time() - start_time > DEADLINE - 20:
            break
        seeds.append(greedy_construct())
    
    # 2. Evaluate seeds
    for s in seeds:
        if time.time() - start_time > DEADLINE - 10:
            break
        if not is_linear(s):
            continue
        a = compute_alpha_fast(s)
        if a < best_alpha:
            best_alpha = a
            best_triples = list(s)
    
    if best_triples is None:
        # Fallback
        best_triples = greedy_construct()
        best_alpha = compute_alpha_fast(best_triples)
    
    # 3. Local Search on the best seed
    if time.time() - start_time < DEADLINE - 5:
        best_triples, best_alpha = local_search(best_triples, time_limit=DEADLINE - 5 - (time.time() - start_time))
    
    # 4. Try local search on other promising seeds if time remains
    for s in seeds:
        if time.time() - start_time > DEADLINE - 5:
            break
        if s == best_triples:
            continue
        # Only try if this seed was close in size or alpha
        # Just try it if we have time
        t_copy = list(s)
        t_copy, a = local_search(t_copy, time_limit=min(5.0, DEADLINE - 5 - (time.time() - start_time)))
        if a < best_alpha:
            best_alpha = a
            best_triples = t_copy
    
    return best_triples
