import random
import itertools

def calculate_independence_number(triples):
    """
    Calculate the size of the largest subset of {0..15} that contains no triple from 'triples'.
    """
    # Precompute masks for triples
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check subsets from largest size down
    # Points are 0..15
    for k in range(16, 0, -1):
        # Generate all subsets of size k
        # Using itertools.combinations
        for subset in itertools.combinations(range(16), k):
            # Check if any triple is contained in this subset
            # Create mask for subset
            sub_mask = 0
            for p in subset:
                sub_mask |= (1 << p)
            
            is_independent = True
            for tm in triple_masks:
                if (tm & sub_mask) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                return k
    return 0

def greedy_construct(seed):
    random.seed(seed)
    points = list(range(16))
    # Generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    random.shuffle(all_triples)
    
    selected_triples = []
    # To check linearity efficiently, we can store for each pair of points, which triple covers it.
    # If a pair is already covered, we cannot add a triple containing that pair.
    # Wait, linearity means any two triples share at most 1 point.
    # This is equivalent to: no pair of points is contained in more than one triple.
    # So we can maintain a set of covered pairs.
    
    covered_pairs = set()
    
    for t in all_triples:
        # Check if all pairs in t are uncovered
        # Pairs in (a,b,c) are (a,b), (a,c), (b,c)
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        
        valid = True
        for p in pairs:
            if p in covered_pairs:
                valid = False
                break
        
        if valid:
            selected_triples.append(list(t))
            for p in pairs:
                covered_pairs.add(p)
                
    return selected_triples

def construct():
    best_triples = []
    best_score = 17 # Max possible is 16, so 17 is infinity
    
    # Run multiple greedy attempts
    for seed in range(100):
        triples = greedy_construct(seed)
        score = calculate_independence_number(triples)
        if score < best_score:
            best_score = score
            best_triples = triples
            
    return best_triples
