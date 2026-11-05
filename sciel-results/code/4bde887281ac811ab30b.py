"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import time

def construct():
    """
    Find a large Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses CSP with bitsets, randomized backtracking, forward checking,
    and dynamic variable ordering (fewest remaining valid partners).
    """
    required = [1, 2, 5, 11]
    max_val = 284
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Bitset for sums. Max sum is 284+284=568.
    # We use Python integers as bitsets.
    
    def get_initial_sums(elements):
        sums = 0
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                sums |= (1 << s)
        return sums

    def is_valid_candidate(x, elements, used_sums):
        """Check if x can be added to elements."""
        # Check 2*x
        if (used_sums >> (2 * x)) & 1:
            return False
        # Check x + e for e in elements
        # We can iterate or use bit tricks. Iteration is fine for small N.
        for e in elements:
            if (used_sums >> (x + e)) & 1:
                return False
        return True

    def get_valid_candidates(elements, used_sums, remaining):
        """Return list of candidates from remaining that are valid to add."""
        valid = []
        for x in remaining:
            if is_valid_candidate(x, elements, used_sums):
                valid.append(x)
        return valid

    def count_valid_partners(c, valid_candidates, used_sums):
        """
        Count how many other candidates d in valid_candidates 
        are compatible with c (i.e., c+d is not in used_sums).
        Note: We assume c and d are both individually valid.
        They conflict if c+d is already in used_sums.
        """
        count = 0
        for d in valid_candidates:
            if d == c:
                continue
            # If c+d is in used_sums, they can't both be in the set.
            if not ((used_sums >> (c + d)) & 1):
                count += 1
        return count

    best_size = len(required)
    best_result = list(required)
    
    # Initial state
    initial_elements = sorted(required)
    initial_sums = get_initial_sums(initial_elements)
    all_candidates = [x for x in range(1, max_val + 1) if x not in required]
    
    # We will use a stack for iterative deepening or just DFS with time limit.
    # Given the depth is small (~16 more to reach 20), recursion is fine.
    
    def search(elements, used_sums, remaining, depth):
        nonlocal best_size, best_result, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if len(elements) > best_size:
            best_size = len(elements)
            best_result = list(elements)
            # If we reached 20, we can stop? The max is 20.
            if best_size >= 20:
                return

        if not remaining:
            return
        
        # Get valid candidates
        valid_candidates = get_valid_candidates(elements, used_sums, remaining)
        
        if not valid_candidates:
            return
        
        # Dynamic Variable Ordering:
        # Choose candidate with fewest remaining valid partners.
        # To add randomness, we can pick among the top few with min partners.
        # Or just pick the min.
        
        # Optimization: If there are many valid candidates, calculating partners for all is O(N^2).
        # We can sample if N is large? N is max 280. 280^2 is ~80k. 
        # Depth ~16. 1.6M ops. Python can do ~10M-100M simple ops/s. 
        # This might be borderline if backtracking is heavy.
        # Let's try it.
        
        # Calculate scores
        scores = []
        for c in valid_candidates:
            partners = count_valid_partners(c, valid_candidates, used_sums)
            scores.append((partners, c))
        
        # Sort by partners (ascending)
        scores.sort()
        
        # Randomize the order of candidates with the same (or similar) score?
        # Or just take the first few.
        # To allow different paths, we can shuffle the candidates with the minimum score.
        min_partners = scores[0][0]
        min_candidates = [c for p, c in scores if p == min_partners]
        
        # Randomly pick one from the most constrained
        random.shuffle(min_candidates)
        
        # Try all of them? Or just the best one?
        # Backtracking implies trying multiple.
        # But trying all min_candidates might be expensive.
        # Let's try the first one (randomized among min) first.
        # If it fails, we backtrack and try the next.
        
        # To be efficient, we iterate through the sorted scores.
        # But we can prune: if we find a solution of size 20, we stop.
        
        for partners, c in scores:
            # If we have already found a solution of size 20, break
            if best_size >= 20:
                return
                
            # Add c
            new_elements = elements + [c]
            new_sums = used_sums
            # Update sums
            new_sums |= (1 << (2 * c))
            for e in elements:
                new_sums |= (1 << (c + e))
                
            new_remaining = [x for x in remaining if x != c]
            
            search(new_elements, new_sums, new_remaining, depth + 1)
            
            # If we found a perfect solution, we can stop searching this branch
            if best_size >= 20:
                return

    # Run search
    search(initial_elements, initial_sums, all_candidates, 0)
    
    # If search didn't find anything better, return best
    result = sorted(best_result)
    
    # Final validation
    valid = True
    sums = set()
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
            
    if not valid or not all(x in result for x in required):
        result = [1, 2, 5, 11]
        
    return result

# EVOLVE-BLOCK-END
