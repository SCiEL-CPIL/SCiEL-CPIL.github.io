import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit

    def can_add(x, selected, used_sums):
        """Check if x can be added to selected (Sidon set) without conflict."""
        if 2 * x in used_sums:
            return False
        for y in selected:
            if x + y in used_sums:
                return False
        return True

    def get_sums(selected):
        """Compute the set of all sums a+b with a<=b from selected."""
        sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                sums.add(selected[i] + selected[j])
        return sums

    def is_maximal(selected):
        """Check if selected is inclusion-maximal."""
        if not selected:
            return False
        used_sums = get_sums(selected)
        selected_set = set(selected)
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # If x can be added, it's not maximal
            if can_add(x, selected, used_sums):
                return False
        return True

    def greedy_construct(randomize=False, bias=0.0):
        """
        Construct a maximal Sidon set using a greedy strategy.
        At each step, pick the element that blocks the most other elements.
        """
        selected = []
        used_sums = set()
        
        while True:
            if time.time() > deadline:
                break
                
            # Find all addable elements
            addable = []
            for x in range(1, N + 1):
                if can_add(x, selected, used_sums):
                    addable.append(x)
            
            if not addable:
                break
            
            if len(addable) == 1:
                x = addable[0]
                selected.append(x)
                used_sums.add(2 * x)
                for y in selected[:-1]:
                    used_sums.add(x + y)
                continue
            
            # Evaluate candidates
            best_x = None
            best_score = -1
            best_ties = []
            
            for x in addable:
                # Simulate adding x
                # New sums added: 2x, x+y for y in selected
                new_sums = set(used_sums)
                new_sums.add(2 * x)
                for y in selected:
                    new_sums.add(x + y)
                
                # Count how many other addable elements are blocked
                blocked_count = 0
                for y in addable:
                    if y == x:
                        continue
                    if not can_add(y, selected + [x], new_sums):
                        blocked_count += 1
                
                score = blocked_count
                if bias != 0.0:
                    if bias > 0:
                        score -= x * bias
                    else:
                        score += x * abs(bias)
                
                if randomize:
                    score += random.uniform(0, 0.1)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_ties = [x]
                elif score == best_score:
                    best_ties.append(x)
            
            # Pick from ties
            if randomize and best_ties:
                x = random.choice(best_ties)
            else:
                x = best_x if best_x is not None else addable[0]
            
            # Add x to selected
            selected.append(x)
            used_sums.add(2 * x)
            for y in selected[:-1]:
                used_sums.add(x + y)
                
        return selected

    def prune(selected):
        """
        Try to remove elements from selected while maintaining maximality.
        Returns the pruned set.
        """
        selected = list(selected)
        improved = True
        while improved:
            improved = False
            # Try removing each element
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline:
                    return selected
                
                # Try removing selected[i]
                new_selected = selected[:i] + selected[i+1:]
                
                if not new_selected:
                    continue
                
                if is_maximal(new_selected):
                    selected = new_selected
                    improved = True
                    break # Restart loop as indices changed
        
        return selected

    # Initial deterministic run
    best_set = greedy_construct(randomize=False)
    best_set = prune(best_set)
    best_size = len(best_set)
    
    # Randomized restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try different randomization strategies
        r = rng.random()
        if r < 0.5:
            current_set = greedy_construct(randomize=True)
        elif r < 0.8:
            # Bias towards smaller numbers (often denser blocking in lower range?)
            current_set = greedy_construct(randomize=True, bias=0.05)
        else:
            # Bias towards larger numbers
            current_set = greedy_construct(randomize=True, bias=-0.05)
        
        current_set = prune(current_set)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we find a very small set, we might want to spend more time
            # but for now, just continue searching.
            
    return best_set
