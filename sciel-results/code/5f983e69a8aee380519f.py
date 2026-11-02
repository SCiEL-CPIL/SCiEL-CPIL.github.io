import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Pre-compute all t-subsets? No, use a set.
    # Map t-subsets to indices for fast array access?
    # Let's use a set of tuples for uncovered.
    
    best_blocks = None
    best_len = float('inf')
    
    # Helper to generate blocks containing a specific t-subset
    # This is done on the fly
    
    # Helper to check if a set of blocks is a valid covering
    def is_valid(blocks):
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        return len(covered) == len(list(itertools.combinations(range(v), t)))
        
    # Helper to remove redundant blocks
    def remove_redundant(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                # Temporarily remove
                remaining = blocks[:i] + blocks[i+1:]
                # Check coverage
                covered = set()
                for b in remaining:
                    covered.update(itertools.combinations(b, t))
                if len(covered) == len(list(itertools.combinations(range(v), t))):
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        return blocks

    # Greedy construction with random pivot
    def greedy_random_pivot():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            pivot_triple = random.choice(list(uncovered))
            
            # Generate all k-subsets containing pivot_triple
            # pivot_triple is a tuple of t points
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            # We need to choose (k - t) points from other_points
            # to form a block with pivot_points
            # Number of such blocks is C(v-t, k-t)
            
            best_block = None
            best_gain = -1
            
            # Iterate over combinations of (k-t) points from other_points
            # For v=24, k=7, t=3: C(21, 4) = 5985
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                
                # Calculate gain: number of uncovered triples in this block
                gain = 0
                for tri in itertools.combinations(block, t):
                    if tri in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we found a block covering all uncovered?
                            # Unlikely, but we can check if gain == len(uncovered) (impossible unless 1 block)
                            pass
                            
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                # Remove covered triples
                for tri in itertools.combinations(best_block, t):
                    if tri in uncovered:
                        uncovered.remove(tri)
                        
        return blocks

    # Run multiple iterations
    iterations = 0
    while time.time() < deadline:
        blocks = greedy_random_pivot()
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Optional: if we reach a very good number, we can try to optimize further
            # but for now, just keep the best.
            
        iterations += 1
        # If we have a good result, we can reduce the time spent on subsequent iterations
        # or just keep going.
        
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            pivot_triple = min(uncovered)
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
        best_blocks = remove_redundant(blocks)

    return best_blocks
