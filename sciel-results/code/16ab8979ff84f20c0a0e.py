import time
import itertools
import random

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
TRIPLE_PAIRS = [set(itertools.combinations(t, 2)) for t in ALL_TRIPLES]
TRIPLE_MASKS = [sum(1 << x for x in t) for t in ALL_TRIPLES]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_alpha = N + 1
    best_triples = []
    
    # Precompute universe mask
    UNIVERSE_MASK = (1 << N) - 1
    
    def calc_alpha_fast(triples):
        """
        Calculates independence number by marking all supersets of edges as dependent.
        Returns the size of the largest independent set.
        """
        if not triples:
            return N
            
        # independent[i] is True if subset i is independent
        # We use a list of booleans or an array. 
        # Size 2^16 = 65536. 
        independent = [True] * (1 << N)
        
        for t in triples:
            m = TRIPLE_MASKS[ALL_TRIPLES.index(t)] # This is slow. Pass masks instead.
            # Optimization: Pass list of masks.
            
        # Let's rewrite to accept masks.
        pass

    def calc_alpha_from_masks(masks):
        independent = [True] * (1 << N)
        
        for m in masks:
            # Iterate all supersets of m
            # Supersets of m are m | z where z is subset of ~m
            # Let's use the trick: 
            # x = m
            # while x < 2**N:
            #    independent[x] = False
            #    x = (x + m) & ~m | m  -- This is wrong.
            
            # Correct superset iteration:
            # Let y = ~m & UNIVERSE_MASK
            # Iterate subsets z of y. Superset is m | z.
            # z = y
            # while True:
            #    independent[m | z] = False
            #    if z == 0: break
            #    z = (z - 1) & y
            
            y = (~m) & UNIVERSE_MASK
            z = y
            while True:
                independent[m | z] = False
                if z == 0:
                    break
                z = (z - 1) & y
        
        # Find max popcount of independent[i]
        max_alpha = 0
        # We can iterate i from 0 to 2^N-1 and check popcount
        # Or iterate by size k downwards.
        # Since N=16, 65536 iterations is trivial.
        for i in range(1 << N):
            if independent[i]:
                # Count bits
                # bin(i).count("1") is fast enough for 65k items?
                # 65k * ~10ns = 0.6ms. 
                # Total for 100 restarts: 60ms. Acceptable.
                c = bin(i).count("1")
                if c > max_alpha:
                    max_alpha = c
        return max_alpha

    def greedy_random(seed_val):
        random.seed(seed_val)
        triples = []
        used_pairs = set()
        masks = []
        
        # Shuffle triples
        indices = list(range(len(ALL_TRIPLES)))
        random.shuffle(indices)
        
        for idx in indices:
            t = ALL_TRIPLES[idx]
            pairs = TRIPLE_PAIRS[idx]
            if pairs.isdisjoint(used_pairs):
                triples.append(t)
                used_pairs |= pairs
                masks.append(TRIPLE_MASKS[idx])
                
        alpha = calc_alpha_from_masks(masks)
        return alpha, triples

    # Run multiple restarts
    # We can also try a "greedy with lookahead" or just many random restarts.
    # Given the knowledge hint, random restarts with fast alpha check work well.
    
    num_restarts = 0
    while time.time() < deadline:
        seed = num_restarts
        alpha, triples = greedy_random(seed)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 4: # Theoretical min likely 4 or 5. If we hit 4, it's great.
                break
        num_restarts += 1
        
    return best_triples
