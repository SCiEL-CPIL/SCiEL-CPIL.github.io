"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Construct a linear triple system on 16 points with small independence number.
    
    The goal is to minimize the independence number (largest set containing no triple).
    We use a greedy approach with backtracking to build a maximal linear system
    (pairwise intersecting in at most one point) that covers points well.
    
    Strategy:
    1. Generate all valid triples on 16 points.
    2. Use a greedy algorithm with randomization to select triples that are
       pairwise intersecting in at most one point (linear system).
    3. Try to maximize coverage to reduce the independence number.
    4. Return the list of triples.
    """
    points = list(range(16))
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    
    def are_compatible(triple1, triple2):
        """Check if two triples intersect in at most one point."""
        intersection = set(triple1) & set(triple2)
        return len(intersection) <= 1
    
    def compute_independence_number(triples):
        """
        Compute the independence number: largest set of points containing no triple.
        For efficiency, we check subsets up to a certain size.
        """
        if not triples:
            return 16
        
        # Convert triples to a set for fast lookup
        triple_set = set(tuple(sorted(t)) for t in triples)
        
        # Try to find the largest independent set
        # We'll check from size 16 down to 1
        for size in range(16, 0, -1):
            # Check if there exists a subset of this size with no triple
            found = False
            for subset in itertools.combinations(points, size):
                subset_set = set(subset)
                # Check if any triple is entirely within this subset
                has_triple = False
                for t in triple_set:
                    if all(p in subset_set for p in t):
                        has_triple = True
                        break
                if not has_triple:
                    found = True
                    break
            if found:
                return size
        return 1
    
    def greedy_construct(seed=42):
        """
        Greedy construction with randomization.
        """
        rng = random.Random(seed)
        
        # Shuffle triples for randomization
        shuffled_triples = all_triples[:]
        rng.shuffle(shuffled_triples)
        
        selected = []
        selected_sets = []
        
        for triple in shuffled_triples:
            # Check compatibility with all selected triples
            compatible = True
            for sel in selected:
                if not are_compatible(triple, sel):
                    compatible = False
                    break
            
            if compatible:
                selected.append(triple)
                selected_sets.append(set(triple))
        
        return selected
    
    # Try multiple seeds to find a good solution
    best_triples = None
    best_score = 17  # Worst case
    
    # Try a range of seeds
    for seed in range(50):
        triples = greedy_construct(seed)
        if not triples:
            continue
        
        # Quick heuristic: count how many points are covered
        covered = set()
        for t in triples:
            covered.update(t)
        
        # The independence number is at least 16 - max_covered if covered is small
        # But we need to compute it properly. For speed, let's use a heuristic first.
        
        # Compute actual independence number
        score = compute_independence_number(triples)
        
        if score < best_score:
            best_score = score
            best_triples = triples
    
    # If we didn't find anything, return a default
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
