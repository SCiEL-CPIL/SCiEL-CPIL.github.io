"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_triples = None
    best_alpha = 17
    
    def calc_alpha(triples):
        if not triples:
            return 16
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
        
        max_alpha = 0
        for mask in range(1 << 16):
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                size = bin(mask).count('1')
                if size > max_alpha:
                    max_alpha = size
        return max_alpha
    
    def greedy_random_pack():
        points = list(range(16))
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)
        
        chosen = []
        pair_used = set()
        
        for t in all_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            can_add = True
            for p1, p2 in pairs:
                key = (min(p1, p2), max(p1, p2))
                if key in pair_used:
                    can_add = False
                    break
            if can_add:
                chosen.append(list(t))
                for p1, p2 in pairs:
                    key = (min(p1, p2), max(p1, p2))
                    pair_used.add(key)
        
        return chosen
    
    # Run random greedy packings
    while time.time() < deadline:
        triples = greedy_random_pack()
        if len(triples) == 0:
            continue
        
        alpha = calc_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 6:
                break
    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
