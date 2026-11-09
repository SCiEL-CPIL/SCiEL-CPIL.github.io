import itertools
import random
import numpy as np

def _all_balls(q, n):
    """Precompute the list of points covered by a ball of radius 1 around each point."""
    # For efficiency, we work with integer encodings
    # Total points: q**n
    # Each ball covers 1 + n*(q-1) points
    # For q=3, n=7: 1 + 7*2 = 15 points per ball
    
    total = q ** n
    balls = []
    
    # Generate all points as tuples first for reference, then encode
    # Actually, let's work directly with encoded integers for speed
    
    def decode(enc):
        """Decode integer to tuple of digits in base q."""
        digits = []
        for _ in range(n):
            digits.append(enc % q)
            enc //= q
        return tuple(digits)
    
    def encode(digits):
        """Encode tuple of digits in base q to integer."""
        result = 0
        for i in range(n):
            result = result * q + digits[i]
        return result
    
    # Precompute balls for each point
    for enc in range(total):
        digits = decode(enc)
        ball = [enc]
        # For each position, try each other digit
        for pos in range(n):
            for d in range(q):
                if d != digits[pos]:
                    new_digits = list(digits)
                    new_digits[pos] = d
                    ball.append(encode(tuple(new_digits)))
        balls.append(frozenset(ball))
    
    return balls

def construct(q, n, radius):
    if radius != 1:
        # Fallback to direct sum for other radii (though task specifies radius=1)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, use all points
            return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    total = q ** n
    
    # Precompute balls
    balls = _all_balls(q, n)
    
    # Convert balls to bitsets for fast operations
    # Each ball as a set of encoded integers
    # Use a boolean array for coverage
    
    # Greedy set cover
    covered = np.zeros(total, dtype=bool)
    code = []
    remaining = total
    
    # For efficiency, track uncovered count per ball
    # Initially, each ball covers len(ball) points, all uncovered
    # We'll maintain a count of how many uncovered points each ball covers
    
    # This is O(total * ball_size) per step if we recompute, which might be slow
    # total=2187, ball_size=15, steps~186
    # 2187 * 15 * 186 ≈ 6 million operations, should be fine in Python with optimization
    
    # Let's use a smarter approach:
    # Maintain uncovered_points as a set
    # For each step, find the ball that covers the most uncovered points
    
    uncovered = set(range(total))
    
    while uncovered:
        best_ball_idx = -1
        best_count = 0
        
        for idx, ball in enumerate(balls):
            count = len(uncovered & ball)
            if count > best_count:
                best_count = count
                best_ball_idx = idx
        
        if best_ball_idx == -1:
            break
        
        code.append(best_ball_idx)
        uncovered -= balls[best_ball_idx]
    
    # Convert encoded indices back to codewords
    def decode(enc):
        digits = []
        for _ in range(n):
            digits.append(enc % q)
            enc //= q
        return digits
    
    codewords = [decode(enc) for enc in code]
    
    # Local improvement: try to remove codewords if coverage is maintained
    # This is a post-processing step to try to reduce the count
    
    # Build coverage structure: for each point, which codewords cover it
    # point_covers[i] = set of codeword indices that cover point i
    
    point_covers = [set() for _ in range(total)]
    for cw_idx, enc in enumerate(code):
        for p in balls[enc]:
            point_covers[p].add(cw_idx)
    
    # Try to remove each codeword
    changed = True
    while changed:
        changed = False
        for cw_idx in range(len(code)):
            enc = code[cw_idx]
            # Check if removing this codeword leaves all points it covers still covered
            removable = True
            for p in balls[enc]:
                if len(point_covers[p]) == 1:
                    removable = False
                    break
            if removable:
                # Remove this codeword
                code.pop(cw_idx)
                for p in balls[enc]:
                    point_covers[p].discard(cw_idx)
                # Adjust indices
                for i in range(len(code)):
                    if i > cw_idx:
                        # The indices in point_covers need adjustment, but we're rebuilding anyway
                        pass
                # Rebuild point_covers since indices changed
                point_covers = [set() for _ in range(total)]
                for new_idx, new_enc in enumerate(code):
                    for p in balls[new_enc]:
                        point_covers[p].add(new_idx)
                changed = True
                break
    
    codewords = [decode(enc) for enc in code]
    
    # Ensure distinctness
    seen = set()
    distinct_codewords = []
    for cw in codewords:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            distinct_codewords.append(cw)
    
    return distinct_codewords
