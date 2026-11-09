"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    def has_3ap(selected_set, x):
        """Check if adding x to selected_set creates a 3-term AP."""
        # Check if x is the middle: a + c = 2x for a,c in selected_set
        # Check if x is the end: a + x = 2b or b + x = 2a for a,b in selected_set
        sel = list(selected_set)
        for i in range(len(sel)):
            a = sel[i]
            # x is middle: need c = 2x - a in selected_set, c != a
            c = 2 * x - a
            if c != a and c != x and c in selected_set:
                return True
            # x is one end: need b = (a + x) / 2 in selected_set
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b != a and b != x and b in selected_set:
                    return True
        return False
    
    # Greedy: add numbers from 1 to 256 if they don't create a 3-AP
    selected = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if not has_3ap(selected, x):
            selected.add(x)
    
    # Try local search: for each selected element, try removing it and see if we can add a better one
    # Actually, since we want to maximize sum of 1/x, and we're already taking the smallest possible,
    # the greedy should be good. Let's try a different ordering to see if we can do better.
    
    # Alternative: try greedy with different strategies
    best_selected = selected
    best_score = sum(1.0/x for x in selected)
    
    # Try: process in order but skip some small numbers to allow more medium numbers?
    # Actually, 1/x is convex, so we prefer small numbers. The greedy taking smallest first is likely optimal.
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
