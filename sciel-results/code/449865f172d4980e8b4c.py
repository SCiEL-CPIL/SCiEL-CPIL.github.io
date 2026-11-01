import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    all_points = list(range(v))
    all_t_sets = [frozenset(comb) for comb in itertools.combinations(all_points, t)]
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        uncovered_set = set(all_t_sets)
        uncovered_list = list(all_t_sets)
        blocks = []
        
        while uncovered_list:
            # Pick a random uncovered t-set
            idx = random.randint(0, len(uncovered_list) - 1)
            T = uncovered_list[idx]
            P = set(T)
            
            while len(P) < k:
                candidates = [x for x in all_points if x not in P]
                gains = {x: 0 for x in candidates}
                
                for U in uncovered_list:
                    diff = U.difference(P)
                    if len(diff) == 1:
                        x = next(iter(diff))
                        if x in gains:
                            gains[x] += 1
                            
                max_gain = max(gains.values())
                if max_gain == 0:
                    x = random.choice(candidates)
                else:
                    max_candidates = [x for x, g in gains.items() if g == max_gain]
                    x = random.choice(max_candidates)
                
                P.add(x)
            
            block = list(P)
            blocks.append(block)
            
            # Remove covered t-sets from list and set
            to_remove = []
            for comb in itertools.combinations(block, t):
                fs = frozenset(comb)
                if fs in uncovered_set:
                    to_remove.append(fs)
            
            for fs in to_remove:
                uncovered_set.discard(fs)
                # Remove from list (inefficient if we do pop by value, but list is small enough?)
                # Actually, removing from list is O(N).
                # Better: just iterate and filter?
                # Or use a dictionary mapping frozenset to index?
                # Given N=11k, and we remove ~50 per block, total removals ~11k.
                # O(N^2) total for list removals might be slow.
                # Let's stick to set for membership and just rebuild list?
                # Or just use set and accept O(N) conversion for random choice?
                # The previous version was fine. Let's revert to set + list conversion for simplicity and correctness.
                pass 

        # Revert to simpler set-based approach for reliability
        uncovered = set(all_t_sets)
        blocks = []
        while uncovered:
            T = random.choice(list(uncovered))
            P = set(T)
            while len(P) < k:
                candidates = [x for x in all_points if x not in P]
                gains = {x: 0 for x in candidates}
                for U in uncovered:
                    diff = U.difference(P)
                    if len(diff) == 1:
                        x = next(iter(diff))
                        if x in gains:
                            gains[x] += 1
                max_gain = max(gains.values())
                if max_gain == 0:
                    x = random.choice(candidates)
                else:
                    max_candidates = [x for x, g in gains.items() if g == max_gain]
                    x = random.choice(max_candidates)
                P.add(x)
            block = list(P)
            blocks.append(block)
            for comb in itertools.combinations(block, t):
                uncovered.discard(frozenset(comb))
        
        cover_count = Counter()
        for b in blocks:
            for comb in itertools.combinations(b, t):
                cover_count[frozenset(comb)] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for comb in itertools.combinations(b, t):
                    fs = frozenset(comb)
                    if cover_count[fs] <= 1:
                        redundant = False
                        break
                if redundant:
                    for comb in itertools.combinations(b, t):
                        cover_count[frozenset(comb)] -= 1
                    blocks.pop(i)
                    changed = True
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if time.time() > deadline:
            break
            
    if best_blocks is None:
        uncovered = set(all_t_sets)
        blocks = []
        while uncovered:
            T = min(uncovered)
            P = set(T)
            while len(P) < k:
                candidates = [x for x in all_points if x not in P]
                gains = {x: 0 for x in candidates}
                for U in uncovered:
                    diff = U.difference(P)
                    if len(diff) == 1:
                        x = next(iter(diff))
                        if x in gains:
                            gains[x] += 1
                max_gain = max(gains.values())
                if max_gain == 0:
                    x = candidates[0]
                else:
                    max_candidates = [x for x, g in gains.items() if g == max_gain]
                    x = max_candidates[0]
                P.add(x)
            block = list(P)
            blocks.append(block)
            for comb in itertools.combinations(block, t):
                uncovered.discard(frozenset(comb))
        best_blocks = blocks
        
    return best_blocks
