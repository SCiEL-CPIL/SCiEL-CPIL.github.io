"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct():
    """
    Find a large Sidon set in [1, 284] containing {1, 2, 5, 11}.
    A Sidon set has all pairwise sums a+b (a<=b) distinct.
    This is equivalent to a Golomb ruler where differences are distinct,
    but here we require sums to be distinct.
    
    We use randomized backtracking with forward checking and dynamic variable ordering.
    """
    # The problem: find max size subset of {1..284} containing {1,2,5,11}
    # such that all pairwise sums are distinct.
    
    # Note: The condition "pairwise sums a+b (a<=b) are all distinct" means:
    # For any two pairs (a,b) and (c,d) with a<=b, c<=d, if a+b == c+d then {a,b}={c,d}.
    # This is the Sidon set property (also called B2 sequence).
    
    # Strategy: Start with the required elements, then greedily/backtrackingly add more.
    # Since the optimal without constraints is 20, we aim for close to that.
    
    # Bitset approach: Represent the set of used sums as a bitset (Python int).
    # Sum ranges from 1+1=2 to 284+284=568. So we need bits 0..568.
    
    required = [1, 2, 5, 11]
    
    # Precompute: for each candidate x, what sums would it create with existing elements?
    # We'll use a randomized backtracking search.
    
    # To make this efficient, we'll try multiple random restarts with different orderings.
    
    best_result = list(required)
    best_size = len(required)
    
    # We'll do randomized backtracking with time limit awareness.
    # Since we need to finish within 120s, we'll be careful.
    
    def get_sums(elements):
        """Get the set of all pairwise sums for a list of elements."""
        sums = set()
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                sums.add(elements[i] + elements[j])
        return sums
    
    def can_add(elements, used_sums, x):
        """Check if x can be added to elements without creating duplicate sums."""
        # New sums: x + e for e in elements, and 2*x
        new_sums = set()
        for e in elements:
            s = x + e
            if s in used_sums:
                return False
            new_sums.add(s)
        s = 2 * x
        if s in used_sums:
            return False
        new_sums.add(s)
        # Check for duplicates among new_sums themselves
        if len(new_sums) != len(elements) + 1:
            return False
        return True
    
    def backtrack(elements, used_sums, candidates, depth, max_depth):
        """
        Recursive backtracking to build a Sidon set.
        elements: current list of selected elements (sorted)
        used_sums: set of all pairwise sums so far
        candidates: list of remaining candidate values
        depth: current depth
        max_depth: maximum depth to search
        """
        nonlocal best_result, best_size
        
        if len(elements) > best_size:
            best_result = list(elements)
            best_size = len(elements)
        
        if depth >= max_depth:
            return
        
        if not candidates:
            return
        
        # Dynamic variable ordering: try candidates that have fewer valid extensions
        # For simplicity, we'll shuffle and try a subset, with some heuristic.
        
        # Try to add each candidate
        for i, x in enumerate(candidates):
            if can_add(elements, used_sums, x):
                new_elements = elements + [x]
                new_sums = used_sums | {x + e for e in elements} | {2 * x}
                new_candidates = candidates[:i] + candidates[i+1:]
                
                backtrack(new_elements, new_sums, new_candidates, depth + 1, max_depth)
                
                # Early termination if we found a good solution
                if best_size >= 19:
                    return
    
    # The search space is too large for pure backtracking over all 280 candidates.
    # Instead, we use a greedy randomized approach with multiple restarts.
    
    def greedy_randomized(num_restarts=50):
        nonlocal best_result, best_size
        
        for _ in range(num_restarts):
            elements = list(required)
            elements.sort()
            used_sums = get_sums(elements)
            
            # Candidates: all numbers from 1 to 284 not in elements
            candidates = [x for x in range(1, 285) if x not in elements]
            
            # Shuffle candidates with some bias toward smaller numbers first
            # (smaller numbers tend to have more constraints)
            random.shuffle(candidates)
            
            current_elements = list(elements)
            current_sums = set(used_sums)
            
            for x in candidates:
                if can_add(current_elements, current_sums, x):
                    current_elements.append(x)
                    current_sums |= {x + e for e in current_elements[:-1]} | {2 * x}
            
            if len(current_elements) > best_size:
                best_result = current_elements
                best_size = len(current_elements)
                
            if best_size >= 19:
                break
    
    # First, try a deterministic greedy to get a baseline
    def greedy_deterministic():
        nonlocal best_result, best_size
        
        elements = list(required)
        elements.sort()
        used_sums = get_sums(elements)
        
        for x in range(1, 285):
            if x in elements:
                continue
            if can_add(elements, used_sums, x):
                elements.append(x)
                used_sums |= {x + e for e in elements[:-1]} | {2 * x}
        
        if len(elements) > best_size:
            best_result = elements
            best_size = len(elements)
    
    # Run deterministic greedy first
    greedy_deterministic()
    
    # Then run randomized restarts
    greedy_randomized(num_restarts=100)
    
    # If we haven't found a good solution, try a more sophisticated approach
    if best_size < 15:
        # Try local search / hill climbing
        def local_search():
            nonlocal best_result, best_size
            
            current = list(best_result)
            current.sort()
            
            for iteration in range(1000):
                improved = False
                # Try to add a random element
                candidates = [x for x in range(1, 285) if x not in current]
                if candidates:
                    random.shuffle(candidates)
                    for x in candidates:
                        if can_add(current, get_sums(current), x):
                            current.append(x)
                            current.sort()
                            improved = True
                            if len(current) > best_size:
                                best_result = list(current)
                                best_size = len(current)
                            break
                
                # Try to swap: remove one element, add another
                if not improved and len(current) > 4:
                    # Remove a random non-required element
                    non_required = [x for x in current if x not in required]
                    if non_required:
                        to_remove = random.choice(non_required)
                        trial = [x for x in current if x != to_remove]
                        trial_sums = get_sums(trial)
                        
                        candidates = [x for x in range(1, 285) if x not in trial]
                        random.shuffle(candidates)
                        for x in candidates:
                            if can_add(trial, trial_sums, x):
                                trial.append(x)
                                trial.sort()
                                if len(trial) > len(current):
                                    current = trial
                                    improved = True
                                    if len(current) > best_size:
                                        best_result = list(current)
                                        best_size = len(current)
                                break
                
                if not improved:
                    break
        
        local_search()
    
    # Ensure the result is valid
    result = sorted(best_result)
    
    # Validate
    sums = set()
    valid = True
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
        # Fallback to a known valid set
        result = [1, 2, 5, 11]
    
    return result

# EVOLVE-BLOCK-END
