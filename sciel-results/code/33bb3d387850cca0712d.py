import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_triples = []
    best_count = 0
    
    def is_valid_addition(triples_set, new_triple):
        x1, y1, z1 = new_triple
        for x2 in range(V):
            if x2 == x1: continue
            for y2 in range(V):
                if y2 == y1 or y2 == x2: continue
                for z2 in range(V):
                    if z2 == z1 or z2 == x2 or z2 == y2: continue
                    
                    if tuple(sorted((x1, y1, z2))) not in triples_set: continue
                    if tuple(sorted((x1, y2, z1))) not in triples_set: continue
                    if tuple(sorted((x1, y2, z2))) not in triples_set: continue
                    if tuple(sorted((x2, y1, z1))) not in triples_set: continue
                    if tuple(sorted((x2, y1, z2))) not in triples_set: continue
                    if tuple(sorted((x2, y2, z1))) not in triples_set: continue
                    if tuple(sorted((x2, y2, z2))) not in triples_set: continue
                    
                    return False
        return True

    def greedy_randomized():
        triples = set()
        shuffled_triples = ALL_TRIPLES[:]
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            if is_valid_addition(triples, t):
                triples.add(t)
        
        return list(triples)

    while time.time() < deadline:
        current_triples = greedy_randomized()
        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_triples = current_triples
            
    return best_triples
