import time
import random
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points maximizing the number of triples
    such that no three disjoint pairs have all eight transversal triples present
    (i.e., avoiding K(2,2,2) configurations).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is formed by three disjoint pairs (a,a'), (b,b'), (c,c')
    # The eight transversal triples are all combinations picking one from each pair
    # We need to find all sets of 8 triples that form such a configuration
    
    # For efficiency, we'll precompute for each triple, which K(2,2,2)s it belongs to
    # But first, let's identify all K(2,2,2)s
    
    # A K(2,2,2) is determined by 6 distinct points partitioned into 3 pairs
    # Let's enumerate all ways to choose 6 points and partition into 3 pairs
    # Number of ways: C(16,6) * 15 = 8008 * 15 = 120120 configurations
    
    # For each configuration, the 8 transversal triples are:
    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # We'll store for each K(2,2,2) the set of 8 triple indices
    # And for each triple, the list of K(2,2,2) indices it belongs to
    
    k222_list = []  # List of sets of 8 triple indices
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    # Enumerate all 6-point subsets and their 15 pairings
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # 15 ways to partition 6 points into 3 pairs
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            # Generate the 8 transversal triples
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            # Convert to sorted tuples for lookup
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    # Greedy algorithm with dynamic weights
    # Start with empty set, add triples that don't complete any K(2,2,2)
    # Use weights to break ties and guide selection
    
    best_result = []
    best_count = 0
    
    # Try multiple random restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Randomize the order of triples
        order = list(range(num_triples))
        random.shuffle(order)
        
        # Initialize weights
        weights = [1.0] * num_triples
        
        # Track which K(2,2,2)s are "threatened" (have some but not all triples present)
        k222_count = [0] * num_k222s  # Number of triples from each K(2,2,2) in current set
        present = set()
        current_set = []
        
        # Process triples in randomized order
        for triple_idx in order:
            if time.time() > deadline - 2:
                break
            
            # Check if adding this triple would complete any K(2,2,2)
            can_add = True
            for k_idx in triple_to_k222s[triple_idx]:
                if k222_count[k_idx] == 7:
                    # Adding this triple would complete the K(2,2,2)
                    can_add = False
                    break
            
            if can_add:
                # Add the triple
                present.add(triple_idx)
                current_set.append(triple_idx)
                
                # Update K(2,2,2) counts
                for k_idx in triple_to_k222s[triple_idx]:
                    k222_count[k_idx] += 1
                
                # Update weights: decrease weight of triples that conflict with this one
                # (i.e., triples in the same K(2,2,2)s)
                for k_idx in triple_to_k222s[triple_idx]:
                    for other_idx in k222_list[k_idx]:
                        if other_idx not in present:
                            weights[other_idx] *= 0.95
        
        # Update best result
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_result = [list(all_triples[i]) for i in current_set]
    
    # If we didn't find anything better, use a simple greedy as fallback
    if best_count == 0:
        present = set()
        k222_count = [0] * num_k222s
        for triple_idx in range(num_triples):
            can_add = True
            for k_idx in triple_to_k222s[triple_idx]:
                if k222_count[k_idx] == 7:
                    can_add = False
                    break
            if can_add:
                present.add(triple_idx)
                for k_idx in triple_to_k222s[triple_idx]:
                    k222_count[k_idx] += 1
        best_result = [list(all_triples[i]) for i in present]
    
    return best_result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
