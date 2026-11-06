import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Generate all K(2,2,2) boxes
    # A box is defined by 3 disjoint pairs
    # For 3 disjoint pairs (a,a'), (b,b'), (c,c'), the 8 transversal triples are:
    # (a,b,c), (a,b,c'), (a',b,c), (a',b,c'), (a,b',c), (a,b',c'), (a',b',c), (a',b',c')
    
    boxes = []  # list of frozensets of triples
    box_of_triple = [[] for _ in range(num_triples)]  # for each triple, list of box indices
    
    # Generate all ways to pick 3 disjoint pairs
    # Pick 6 distinct points, partition into 3 pairs
    for six_points in itertools.combinations(range(n), 6):
        # Generate all partitions of 6 points into 3 pairs
        a, b, c, d, e, f = six_points
        # All pairings: fix a's partner, then pair the rest
        pairings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in pairings:
            # Each pair is (x, x'), we need to assign which is the "first" and "second"
            # But the 8 transversal triples don't depend on the ordering within pairs
            # Let's just compute the 8 triples
            a1, a2 = p1
            b1, b2 = p2
            c1, c2 = p3
            
            trans_triples = [
                (a1, b1, c1),
                (a1, b1, c2),
                (a1, b2, c1),
                (a1, b2, c2),
                (a2, b1, c1),
                (a2, b1, c2),
                (a2, b2, c1),
                (a2, b2, c2),
            ]
            
            # Normalize each triple (sort)
            norm_triples = [tuple(sorted(t)) for t in trans_triples]
            
            # Check they're all distinct (they should be since pairs are disjoint)
            if len(set(norm_triples)) == 8:
                box = frozenset(norm_triples)
                box_idx = len(boxes)
                boxes.append(box)
                for t in norm_triples:
                    box_of_triple[triple_index[t]].append(box_idx)
    
    num_boxes = len(boxes)
    # print(f"Generated {num_boxes} boxes")
    
    if time.time() > deadline:
        # Fallback: just return a simple valid set
        return [list(t) for t in all_triples[:100]]
    
    # Now use greedy approach:
    # Start with all triples present
    # Repeatedly find a box that is "full" (all 8 triples present) and remove one triple
    # Choose the triple to remove that hits the most other full boxes
    
    present = [True] * num_triples
    
    def is_box_full(box_idx):
        box = boxes[box_idx]
        for t in box:
            if not present[triple_index[t]]:
                return False
        return True
    
    # Count how many full boxes each triple is in
    # We'll iterate: find full boxes, pick a triple from a full box that's in the most full boxes, remove it
    
    # First, let's count for each triple how many boxes it's in
    triple_box_count = [len(box_of_triple[i]) for i in range(num_triples)]
    
    # Greedy removal
    # We'll maintain a list of full boxes
    # For efficiency, let's just iterate
    
    # Alternative: randomize the order and try to add triples
    # Start empty, add triples one by one in random order, skip if it completes a box
    
    # Let's try the "add" approach with randomization
    best_count = 0
    best_result = []
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        random.seed(trial)
        present = [False] * num_triples
        # box_full_count[box_idx] = number of triples in box that are present
        box_full_count = [0] * num_boxes
        
        # Shuffle triples
        order = list(range(num_triples))
        random.shuffle(order)
        
        count = 0
        for t_idx in order:
            t = all_triples[t_idx]
            # Check if adding this triple completes any box
            can_add = True
            for b_idx in box_of_triple[t_idx]:
                if box_full_count[b_idx] == 7:  # all other 7 are present
                    can_add = False
                    break
            
            if can_add:
                present[t_idx] = True
                count += 1
                for b_idx in box_of_triple[t_idx]:
                    box_full_count[b_idx] += 1
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
    
    return best_result
