import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # 1. Find several minimal Golomb rulers of order k+1 to use as starting rows.
    # A Golomb ruler has distinct differences.
    # We want to find rulers with the smallest possible span.
    
    def find_rulers(k, max_rulers=10):
        """
        Backtracking search for Golomb rulers of length k+1.
        Returns a list of rulers (lists of integers).
        """
        rulers = []
        used_diffs = set()
        current = [0]
        
        def backtrack(idx, last_val, max_val):
            if len(rulers) >= max_rulers:
                return
            if idx == k:
                # Found a ruler
                rulers.append(current[:])
                return
            
            # Try to find the next mark
            # We can prune if the current span is already larger than known bests, 
            # but for finding *several* minimal ones, we just search.
            
            # Heuristic: Try values close to last_val first? 
            # Actually, for minimal span, we should try small increments.
            
            # Symmetry breaking: The next mark must be > last_val.
            # We can limit the search space.
            
            # To find minimal rulers, we should search in increasing order of span.
            # This is hard to do directly in a simple DFS.
            # Instead, we can just do a DFS that finds *some* rulers, 
            # then filter by span.
            
            # Better: IDDFS (Iterative Deepening DFS) on the span?
            # Or just a DFS that tries small steps first.
            
            # Let's try a DFS that tries values from last_val + 1 upwards.
            # To ensure we find minimal ones, we should stop early if we find enough.
            
            # Optimization: If we have already found a ruler with span S, 
            # and current last_val is such that we can't beat S, prune.
            
            # For now, simple DFS with a limit on depth/value.
            
            # To find *minimal* rulers, we should search for the smallest possible next value.
            # If we always pick the smallest valid next value, we get one ruler.
            # To get others, we need to branch.
            
            # Let's try values from last_val + 1 to last_val + 10 (or so).
            # If k is small, this is fast.
            
            # Actually, for k=6 (7 marks), the minimal span is 25.
            # We can just hardcode or quickly find the minimal span first, 
            # then find all rulers with that span.
            
            pass # Placeholder for logic below

        # Let's implement a specific search for minimal span rulers.
        # 1. Find minimal span S_min for order k+1.
        # 2. Find all rulers with span S_min.
        
        # For small k, we can just use a known list or search.
        # Let's do a search for minimal span.
        
        def find_min_span(k):
            # Binary search on span? Or just increase span.
            # For k=6, span is 25.
            # For k=5, span is 17.
            # For k=4, span is 11.
            # We can just try spans starting from a lower bound.
            
            # Lower bound: k*(k+1)/2 is the number of diffs. 
            # The smallest distinct positive integers are 1..N.
            # Sum of first N integers is N(N+1)/2.
            # This doesn't directly give span.
            
            # Let's just use a greedy to find a good span, then search around it.
            # Or just use the greedy result as a candidate.
            
            pass

        # Given the complexity, let's stick to a robust Greedy with Randomization and Local Search.
        # The "Start with minimal ruler" idea is good but implementing a full Golomb ruler solver 
        # inside the 120s limit for arbitrary k might be risky if k is large.
        # However, the task says "7 rows of 6 increasing integers", so k=6 is fixed for this instance?
        # "The evaluator passes this task's n_rows and k." 
        # It implies k might vary.
        
        # Let's assume k is small (e.g., < 10).
        
        # Strategy:
        # 1. Use a greedy algorithm to construct rows.
        # 2. To improve, use a randomized greedy: when multiple choices are available, 
        #    pick one randomly from the top few smallest valid choices.
        # 3. Run multiple iterations, keep the best.
        
        pass

    # Implementation of Randomized Greedy
    
    def greedy_construct(n_rows, k, randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find valid next values
                # We want to pick x such that all diffs x-a for a in row are not in used.
                
                # To speed up, we can just increment x and check.
                # If randomize, we might look at a window of candidates.
                
                candidates = []
                # Look ahead a bit?
                # Let's just find the first few valid x's.
                
                # Optimization: 
                # Instead of checking every x, we can maintain a set of "blocked" x's?
                # No, the blocked x's depend on the current row.
                
                # Just iterate x.
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        if not randomize:
                            break
                        if len(candidates) >= 5:
                            break
                        # Keep looking for a few more candidates if randomizing
                        if x > 200: # Safety break
                            break
                
                if not candidates:
                    # Should not happen if we have enough space
                    break
                
                if randomize:
                    # Pick a random candidate from the top few
                    # Weight towards smaller x
                    # Simple: pick the smallest with high prob
                    if random.random() < 0.8:
                        next_x = candidates[0]
                    else:
                        next_x = random.choice(candidates)
                else:
                    next_x = candidates[0]
                
                row.append(next_x)
                used.update({next_x - a for a in row[:-1]}) # Update with diffs to previous elements
                # Note: diffs to 0 are included in row[:-1] if row was [0]
                # Actually, row is [0, ..., next_x_prev].
                # Diffs are next_x - a for all a in row.
                # We added next_x to row.
                # So diffs are next_x - a for a in row[:-1].
                # Correct.
                
            rows.append(row)
            
        return rows

    best_rows = None
    best_score = float('inf')
    
    # Run deterministic greedy first
    rows = greedy_construct(n_rows, k, randomize=False)
    max_val = max(max(r) for r in rows)
    best_rows = rows
    best_score = max_val
    
    # Run randomized greedy
    # Budget remaining
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run several randomized attempts
        # Number of attempts depends on time
        attempts = 0
        while time.time() < deadline - 2:
            seed = random.randint(0, 1000000)
            rows = greedy_construct(n_rows, k, randomize=True, seed=seed)
            max_val = max(max(r) for r in rows)
            if max_val < best_score:
                best_score = max_val
                best_rows = rows
                # If we found a very good one, we might stop early?
                # No, keep searching.
            attempts += 1
            
    return best_rows

# EVOLVE-BLOCK-END
