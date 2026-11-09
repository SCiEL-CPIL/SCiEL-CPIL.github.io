"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_pairs(family):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        si = set(family[i])
        li = len(si)
        for j in range(i + 1, n):
            sj = set(family[j])
            lj = len(sj)
            if li < lj:
                if si.issubset(sj):
                    count += 1
            elif lj < li:
                if sj.issubset(si):
                    count += 1
    return count

def _is_valid(family):
    """Check if family is valid: 40 distinct subsets of {0..11}."""
    n = P['n']
    m = P['m']
    if len(family) != m:
        return False
    seen = set()
    for s in family:
        if not isinstance(s, (list, tuple)):
            return False
        if len(s) != len(set(s)):
            return False
        if any(v < 0 or v >= n for v in s):
            return False
        key = frozenset(s)
        if key in seen:
            return False
        seen.add(key)
    return True

def _generate_chain_initial():
    """Generate an initial family based on a chain structure."""
    n = P['n']
    m = P['m']
    random.seed(42)
    
    # Create a base chain of 12 sets: sizes 0, 1, ..., 11
    # We need 40 sets. Let's pick 40 sets that form many subset relations.
    # Strategy: Pick a random permutation of vertices, build a chain of 12 sets.
    # Then, for each set in the chain, we can add variations? No, must be distinct.
    # Better: Use the fact that if we have a chain A0 < A1 < ... < A11, 
    # any pair (Ai, Aj) with i < j is comparable.
    # If we just take the chain, we have 12 sets. We need 40.
    # We can add sets that are subsets of the largest or supersets of the smallest?
    # Actually, any set is comparable with any set in the chain if it fits in between?
    # No, a set X is comparable with Ai if X < Ai or X > Ai or X = Ai.
    # If we pick 40 random sets, they might not be comparable with the chain.
    
    # Let's try a different initial construction.
    # Pick 40 sets from the 4096 possible subsets.
    # To maximize comparable pairs, we want a "dense" region in the subset lattice.
    # A good heuristic: pick sets of similar sizes? No, similar sizes are less likely to be comparable.
    # Pick sets of very different sizes? 
    # If we pick 20 small sets (size 0-2) and 20 large sets (size 9-11), 
    # small sets are subsets of large sets? Not necessarily.
    # If S is small and T is large, S < T is possible but not guaranteed.
    # However, if we pick S from a small set of vertices and T from a large set of vertices...
    
    # Let's stick to the chain idea but expand it.
    # Take a random permutation p of 0..11.
    # Chain: C_k = {p[0], ..., p[k-1]} for k=0..11.
    # These 12 sets are mutually comparable (11*12/2 = 66 pairs).
    # Now we need 28 more sets.
    # If we add sets that are subsets of C_11 (which is all vertices), they are not necessarily comparable with other chain members.
    # But if we add sets that are subsets of C_6 (middle), they are comparable with C_0..C_6 and not with C_7..C_11?
    # Actually, any subset of C_k is comparable with C_j for j>=k (since subset of C_k is subset of C_j).
    # And it is comparable with C_j for j<k only if it contains C_j? No, C_j is subset of it?
    # Let S be a subset of C_k.
    # For j >= k: S is subset of C_j. So (S, C_j) is comparable.
    # For j < k: C_j is subset of C_k. Is C_j subset of S? Or S subset of C_j?
    # If S is "large" within C_k, it might contain C_j.
    
    # Let's just generate 40 random sets first, then improve.
    family = []
    seen = set()
    for _ in range(m):
        while True:
            size = random.randint(0, n)
            s = set(random.sample(range(n), size))
            if frozenset(s) not in seen:
                seen.add(frozenset(s))
                family.append(list(s))
                break
    return family

def _local_search(family, time_limit):
    """Improve the family using local search."""
    family = [list(s) for s in family]
    current_score = _count_pairs(family)
    n = P['n']
    m = P['m']
    all_vertices = set(range(n))
    
    while time.time() < time_limit:
        improved = False
        # Try replacing each set with a random one or a related one
        for i in range(m):
            if time.time() >= time_limit:
                break
            
            old_s = family[i]
            old_set = set(old_s)
            
            # Generate candidate replacements
            candidates = []
            
            # 1. Random set of random size
            for _ in range(5):
                size = random.randint(0, n)
                candidates.append(set(random.sample(range(n), size)))
            
            # 2. Subset of old set
            if len(old_set) > 0:
                size = random.randint(0, len(old_set))
                candidates.append(set(random.sample(list(old_set), size)))
            
            # 3. Superset of old set
            available = list(all_vertices - old_set)
            if available:
                size = random.randint(1, len(available))
                candidates.append(old_set | set(random.sample(available, size)))
            
            # 4. Random set close in size
            size = max(0, min(n, len(old_set) + random.randint(-2, 2)))
            candidates.append(set(random.sample(range(n), size)))
            
            best_candidate = None
            best_score = current_score
            
            for cand in candidates:
                if frozenset(cand) == frozenset(old_set):
                    continue
                # Check distinctness
                is_dup = False
                for j in range(m):
                    if j != i and frozenset(cand) == frozenset(family[j]):
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Evaluate
                family[i] = list(cand)
                new_score = _count_pairs(family)
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = list(cand)
            
            if best_candidate is not None:
                family[i] = best_candidate
                current_score = best_score
                improved = True
            else:
                family[i] = old_s
        
        if not improved:
            # Try swapping two sets? Swapping doesn't change the set of sets, so score is same.
            # Try mutating two sets simultaneously?
            # Or restart?
            # Let's try a more aggressive mutation: pick a random set and replace it with a subset of a random other set
            i = random.randint(0, m - 1)
            j = random.randint(0, m - 1)
            if i != j:
                base_set = set(family[j])
                size = random.randint(0, len(base_set))
                cand = set(random.sample(list(base_set), size))
                
                is_dup = False
                for k in range(m):
                    if k != i and frozenset(cand) == frozenset(family[k]):
                        is_dup = True
                        break
                if not is_dup:
                    old_s = family[i]
                    family[i] = list(cand)
                    new_score = _count_pairs(family)
                    if new_score > current_score:
                        current_score = new_score
                    else:
                        family[i] = old_s

    return family, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # Try a few different initial strategies
    best_family = None
    best_score = -1
    
    # Strategy 1: Random
    random.seed(1)
    family1 = _generate_chain_initial()
    f1, s1 = _local_search(family1, deadline - 10)
    if s1 > best_score:
        best_score = s1
        best_family = f1
    
    # Strategy 2: Chain-based initial
    # Create a chain and fill the rest with subsets/supersets
    random.seed(2)
    perm = list(range(P['n']))
    random.shuffle(perm)
    chain = [set(perm[:k]) for k in range(P['n'] + 1)]
    family2 = [list(s) for s in chain]
    seen = set(frozenset(s) for s in family2)
    
    # Add 28 more sets
    # Try to add sets that are subsets of the large chain members or supersets of small ones
    # Or just random
    while len(family2) < P['m']:
        size = random.randint(0, P['n'])
        s = set(random.sample(range(P['n']), size))
        if frozenset(s) not in seen:
            seen.add(frozenset(s))
            family2.append(list(s))
    
    f2, s2 = _local_search(family2, deadline - 5)
    if s2 > best_score:
        best_score = s2
        best_family = f2
    
    # If time remains, do more local search on the best one
    if best_family is not None and time.time() < deadline:
        f_best, s_best = _local_search(best_family, deadline)
        if s_best > best_score:
            best_score = s_best
            best_family = f_best
    
    # Fallback
    if best_family is None:
        best_family = [[b for b in range(P['n']) if (mask >> b) & 1] for mask in range(P['m'])]
    
    # Ensure proper types
    result = []
    for s in best_family:
        result.append([int(v) for v in s])
    
    return result
# EVOLVE-BLOCK-END
