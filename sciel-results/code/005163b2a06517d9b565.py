import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill
    
    best = []
    
    def can_add(seq, sums, x):
        """Check if x can be appended to seq without sum collision."""
        # Compute all new consecutive sums ending at x
        s = 0
        for a in reversed(seq):
            s += a
            if s in sums:
                return False
        if x in sums:
            return False
        return True
    
    def new_sums(seq, sums, x):
        """Compute the set of new consecutive sums when appending x."""
        ns = set()
        s = 0
        for a in reversed(seq):
            s += a
            ns.add(s)
        ns.add(x)
        return ns
    
    def dfs(seq, sums, depth):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(seq) > len(best):
            best = seq[:]
        
        # Try to extend
        # Candidate values: greater than last element, up to 256
        last = seq[-1] if seq else 0
        start_val = last + 1
        
        # For efficiency, try values in order but with some heuristics
        # Try smaller gaps first (denser sequences tend to be longer)
        candidates = []
        for x in range(start_val, 257):
            if can_add(seq, sums, x):
                candidates.append(x)
        
        if not candidates:
            return
        
        # Try candidates - use a mix of strategies
        # First try the smallest valid (greedy-like), then explore others
        # But to find longer sequences, we need to explore alternatives
        
        # Limit branching: try at most a few candidates per level
        # Use a heuristic: prefer values that leave more room
        for x in candidates:
            if time.time() > deadline:
                return
            
            ns = new_sums(seq, sums, x)
            new_sums_set = sums | ns
            
            dfs(seq + [x], new_sums_set, depth + 1)
            
            # If we found a longer sequence, we can stop early at this level
            # (since we're trying in order and want max length)
            # Actually, we should continue to see if an even longer one exists
            # But with time limits, let's be selective
    
    # Start DFS
    dfs([], set(), 0)
    
    return best

# Test with a quick check
if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
    
    # Verify
    sums = set()
    valid = True
    for i in range(len(result)):
        s = 0
        for j in range(i, len(result)):
            s += result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
    print(f"Strictly increasing: {all(result[i] < result[i+1] for i in range(len(result)-1))}")
    print(f"All in [1,256]: {all(1 <= x <= 256 for x in result)}")
