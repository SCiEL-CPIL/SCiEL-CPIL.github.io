import time
import sys

def construct():
    deadline = time.time() + 85  # 85 second deadline (5s safety margin)
    n = 128
    
    # Start with all elements
    S = set(range(1, n + 1))
    
    def has_conflict(S):
        """Check if S contains a conflict triple. Return the triple if found."""
        s = sorted(S)
        for i in range(len(s)):
            a = s[i]
            for j in range(i + 1, len(s)):
                b = s[j]
                for k in range(j + 1, len(s)):
                    c = s[k]
                    ab = a + b
                    ac = a + c
                    bc = b + c
                    if ab in S and ac in S and bc in S:
                        # Check all 6 are distinct (a,b,c distinct already)
                        # Need ab, ac, bc distinct from each other and from a,b,c
                        # Since a,b,c >= 1, ab > a, ab > b (if b>0), etc.
                        # Actually ab = a+b > max(a,b) so ab not in {a,b,c} if c < ab
                        # But we need to check ab, ac, bc are distinct from each other
                        # ab = ac implies b = c (impossible since distinct)
                        # So ab, ac, bc are always distinct
                        # Also ab > a, ab > b, so ab not equal to a or b
                        # ab could equal c? ab = a+b = c. Then ac = a+c = a+a+b = 2a+b, bc = b+c = b+a+b = a+2b
                        # So we need to check that ab, ac, bc are not equal to a, b, or c
                        vals = {a, b, c, ab, ac, bc}
                        if len(vals) == 6:
                            return (a, b, c)
        return None
    
    # Greedy peeling
    max_iters = 200
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find a conflict
        conflict = has_conflict(S)
        if conflict is None:
            break
        
        # Count conflicts per element
        # For efficiency, just remove one element from the found conflict
        # Better: find the element in the most conflicts
        # For now, remove the largest element in the conflict (heuristic: larger elements 
        # in lower part cause more issues)
        
        a, b, c = conflict
        # Remove the element that appears in the most potential conflicts
        # Simple heuristic: remove the smallest of a,b,c (to keep upper elements)
        # Or remove the one that's in the "middle" range
        # Let's try: remove the element with the most pair-sum partners in S
        
        candidates = [a, b, c]
        # For each candidate, count how many pairs (x,y) in S\{candidate} 
        # such that x+y, x+cand, y+cand... this is expensive
        # Simpler: just remove the smallest element
        to_remove = min(candidates)
        S.discard(to_remove)
    
    # Convert to list of Python ints
    result = [int(x) for x in sorted(S)]
    return result
