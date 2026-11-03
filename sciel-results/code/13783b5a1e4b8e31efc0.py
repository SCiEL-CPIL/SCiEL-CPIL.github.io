<<<<<<< SEARCH
def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples
       to maximize the reduction in the entropy of the uncovered triple distribution.
    2. Local Search: Optimize the initial solution by removing redundant blocks
       and replacing blocks with more efficient ones.
    """
    all_triples = set(itertools.combinations(range(v), t))
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy(v, k, t, all_triples)
    
    # Phase 2: Local Search Optimization
    blocks = _local_search(v, k, t, all_triples, blocks)
    
    return blocks
=======
def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Greedy Construction with Randomization: Maximizes coverage of uncovered triples.
    2. Local Search: Removes redundant blocks and attempts block replacements.
    3. Multiple Restarts: Keeps the best solution from multiple runs.
    """
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple iterations to find a better solution
    for i in range(20):
        random.seed(i)
        blocks = _info_greedy(v, k, t)
        blocks = _local_search(v, k, t, blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    return best_blocks
>>>>>>> REPLACE

