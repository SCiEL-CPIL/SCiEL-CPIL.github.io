"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute sum conflicts.
    # For a given set S with sums in Sums, a candidate v is valid iff:
    # 1. 2v not in Sums
    # 2. For all s in S: v+s not in Sums
    
    def is_valid_addition(v, selected, sums):
        if 2 * v in sums:
            return False
        for x in selected:
            if (v + x) in sums:
                return False
        return True
    
    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums
    
    def is_maximal(selected, sums):
        """Check if no missing element can be added."""
        selected_set = set(selected)
        for v in range(1, N + 1):
            if v in selected_set:
                continue
            if is_valid_addition(v, selected, sums):
                return False
        return True
    
    def greedy_construct(seed=None):
        """Greedy construction with randomized tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        while available:
            valid_candidates = []
            for v in available:
                if is_valid_addition(v, selected, sums):
                    valid_candidates.append(v)
            
            if not valid_candidates:
                break
            
            # Pick randomly from valid candidates (or use a heuristic)
            # To get diverse sets, just pick randomly
            v = random.choice(valid_candidates)
            
            selected.append(v)
            sums.add(2 * v)
            for x in selected:
                if x != v:
                    sums.add(v + x)
            
            available.remove(v)
        
        return selected, sums
    
    def try_remove_to_minimize(selected, sums):
        """Try to remove elements one at a time if the set remains maximal."""
        selected = list(selected)
        sums = get_sums(selected)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(selected)):
                # Try removing selected[i]
                new_selected = selected[:i] + selected[i+1:]
                new_sums = get_sums(new_selected)
                
                if is_maximal(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    improved = True
                    break
        
        return selected, sums
    
    # Main search
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Generate diverse greedy sets and try to minimize each
    attempts = 0
    max_attempts = 200
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        selected, sums = greedy_construct(seed=None)
        
        # Try to minimize
        minimized_selected, minimized_sums = try_remove_to_minimize(selected, sums)
        
        if len(minimized_selected) < best_size:
            best_size = len(minimized_selected)
            best_set = minimized_selected
        
        # Early stop if we hit theoretical lower bound
        if best_size <= 16:
            break
    
    # Phase 2: Local search - swap elements to try to get smaller
    if best_set is not None and time.time() < deadline:
        current_set = list(best_set)
        current_sums = get_sums(current_set)
        
        for _ in range(50):
            if time.time() >= deadline:
                break
            
            # Try to find a replacement that allows removing an element
            for i in range(len(current_set)):
                if time.time() >= deadline:
                    break
                
                # Try removing current_set[i]
                new_selected = current_set[:i] + current_set[i+1:]
                
                # Now try to add a random element to see if we can still be maximal
                # Actually, let's try a different approach:
                # Pick a random element to remove, then try to add random elements
                # until maximal again, and check if size decreased
                
                removed = current_set[i]
                base_selected = current_set[:i] + current_set[i+1:]
                base_sums = get_sums(base_selected)
                
                # Try to extend to maximal with random additions
                extended = list(base_selected)
                ext_sums = get_sums(extended)
                
                # Add random valid elements
                available = set(range(1, N + 1)) - set(extended)
                random.shuffle(list(available))
                
                for v in list(available):
                    if v not in available:
                        continue
                    if is_valid_addition(v, extended, ext_sums):
                        extended.append(v)
                        ext_sums.add(2 * v)
                        for x in extended:
                            if x != v:
                                ext_sums.add(v + x)
                        available.remove(v)
                
                # Check if maximal
                if is_maximal(extended, ext_sums):
                    if len(extended) < len(current_set):
                        current_set = extended
                        current_sums = ext_sums
            
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = list(current_set)
    
    # Fallback
    if best_set is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if is_valid_addition(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for y in selected:
                    if y != x:
                        sums.add(x + y)
        best_set = selected
    
    return best_set
# EVOLVE-BLOCK-END
