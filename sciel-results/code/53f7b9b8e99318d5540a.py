"""Rewritten program: beam search over coprime sets to maximize sum 1/(512-a)."""
import time
from math import gcd
from fractions import Fraction
import heapq
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    N = 512
    
    def score_set(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += Fraction(1, N - a)
        return s
    
    def greedy_seed():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_set = greedy_seed()
    best_score = score_set(best_set)
    
    # Precompute float weights for fast beam operations
    float_weights = [1.0 / (N - a) for a in range(1, 512)]
    
    # Precompute coprime compatibility using bitsets for speed
    # coprime_mask[a] is a bitmask of which values 1..511 are coprime with a
    # We'll use a list of sets for clarity, then optimize if needed
    coprime_with = [None] * 512
    for a in range(1, 512):
        coprime_with[a] = set()
        for b in range(1, 512):
            if gcd(a, b) == 1:
                coprime_with[a].add(b)
    
    # Beam search with float scores for speed
    beam_width = 2000
    # Each beam entry: (float_score, chosen_tuple)
    beam = [(0.0, ())]
    
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        
        new_beam = []
        w = float_weights[a - 1]  # 1/(512-a)
        
        for score, chosen in beam:
            # Option 1: Skip a
            new_beam.append((score, chosen))
            
            # Option 2: Include a (if coprime with all in chosen)
            if all(b in coprime_with[a] for b in chosen):
                new_score = score + w
                new_chosen = chosen + (a,)
                new_beam.append((new_score, new_chosen))
        
        # Keep top beam_width entries by float score
        if len(new_beam) > beam_width:
            new_beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
        
        beam = new_beam
    
    # Find the best complete solution in the final beam using exact score
    for score, chosen in beam:
        exact_score = score_set(chosen)
        if exact_score > best_score:
            best_score = exact_score
            best_set = list(chosen)
    
    # Local search polish: try to swap elements to improve
    def local_search_improve(chosen_list):
        """Try swapping elements to improve the score."""
        chosen_set = set(chosen_list)
        current_score = score_set(chosen_list)
        improved = True
        max_iterations = 5000
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() < deadline - 5:
            improved = False
            iteration += 1
            
            for i, a in enumerate(chosen_list):
                # Try removing a
                new_list = chosen_list[:i] + chosen_list[i+1:]
                new_score = score_set(new_list)
                if new_score > current_score:
                    chosen_list = new_list
                    current_score = new_score
                    improved = True
                    break
                
                # Try swapping a with b not in chosen
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    # Check if b is coprime with all other elements
                    others = [x for x in chosen_list if x != a]
                    if all(gcd(b, x) == 1 for x in others):
                        new_list = chosen_list[:i] + [b] + chosen_list[i+1:]
                        new_score = score_set(new_list)
                        if new_score > current_score:
                            chosen_set.discard(a)
                            chosen_set.add(b)
                            chosen_list = new_list
                            current_score = new_score
                            improved = True
                            break
                
                if improved:
                    break
        
        return chosen_list, current_score
    
    # Apply local search to the best solution
    if time.time() < deadline - 10:
        polished_list, polished_score = local_search_improve(list(best_set))
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_list
    
    return list(best_set)

# Let's test with a quick run
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Size: {len(result)}")
    from fractions import Fraction
    s = sum(Fraction(1, 512-a) for a in result)
    print(f"Score: {float(s)}")
