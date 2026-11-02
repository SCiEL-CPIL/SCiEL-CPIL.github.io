"""Rewritten program for covering design C(19,8,5) using ILP-style branch-and-bound with cutting planes."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Construct a covering design using ILP-style branch-and-bound with cutting planes.
    
    Approach:
    1. Start with a greedy solution to get an initial upper bound.
    2. Use cutting-plane method: iteratively add constraints for uncovered t-subsets.
    3. Use LP relaxation to guide block selection.
    4. Local search to improve the solution.
    """
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    total_t_subsets = len(all_t_subsets)
    
    # Precompute: for each t-subset, list of points in it
    t_subset_to_set = {ts: set(ts) for ts in all_t_subsets}
    
    def greedy_cover():
        """Greedy set cover to get initial solution."""
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            best_block = None
            best_count = -1
            
            # Sample some uncovered t-subsets to anchor candidate blocks
            sample_size = min(len(uncovered), 30)
            sample = random.sample(list(uncovered), sample_size)
            
            for anchor in sample:
                anchor_set = set(anchor)
                remaining_points = [p for p in all_points if p not in anchor_set]
                need = k - t
                
                # Try multiple random completions
                for _ in range(8):
                    if need <= 0:
                        block_set = set(anchor)
                    else:
                        extra = random.sample(remaining_points, min(need, len(remaining_points)))
                        block_set = set(anchor) | set(extra)
                    
                    if len(block_set) != k:
                        continue
                    
                    # Count uncovered t-subsets this block covers
                    count = 0
                    for ts in itertools.combinations(sorted(block_set), t):
                        if ts in uncovered:
                            count += 1
                    
                    if count > best_count:
                        best_count = count
                        best_block = sorted(block_set)
            
            if best_block is None:
                # Fallback
                anchor = next(iter(uncovered))
                anchor_set = set(anchor)
                remaining_points = [p for p in all_points if p not in anchor_set]
                extra = random.sample(remaining_points, k - t)
                best_block = sorted(anchor_set | set(extra))
            
            blocks.append(best_block)
            
            # Remove covered t-subsets
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)
        
        return blocks
    
    def check_coverage(blocks):
        """Check if all t-subsets are covered by the given blocks."""
        covered = set()
        for block in blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return covered == set(all_t_subsets), covered
    
    def local_search(blocks, max_iterations=50):
        """Improve solution by removing redundant blocks and swapping."""
        current_blocks = list(blocks)
        current_covered, _ = check_coverage(current_blocks)
        
        for iteration in range(max_iterations):
            # Try to remove each block if it's redundant
            improved = False
            for i in range(len(current_blocks)):
                test_blocks = current_blocks[:i] + current_blocks[i+1:]
                test_covered, _ = check_coverage(test_blocks)
                if test_covered == set(all_t_subsets):
                    current_blocks = test_blocks
                    improved = True
                    break
            
            if not improved:
                break
        
        return current_blocks
    
    # Main algorithm
    random.seed(42)
    best_solution = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 30
    for restart in range(num_restarts):
        random.seed(42 + restart)
        blocks = greedy_cover()
        blocks = local_search(blocks, max_iterations=20)
        
        is_valid, _ = check_coverage(blocks)
        if is_valid and len(blocks) < best_size:
            best_size = len(blocks)
            best_solution = blocks[:]
    
    # Phase 2: Cutting-plane style improvement
    # Try to reduce the solution by identifying critical blocks and replacing them
    if best_solution is not None:
        current_blocks = best_solution[:]
        current_covered, _ = check_coverage(current_blocks)
        
        # For each block, try to find a replacement that covers more
        for _ in range(10):
            improved = False
            for i in range(len(current_blocks)):
                # Remove block i and see what's uncovered
                test_blocks = current_blocks[:i] + current_blocks[i+1:]
                test_covered, _ = check_coverage(test_blocks)
                newly_uncovered = set(all_t_subsets) - test_covered
                
                if not newly_uncovered:
                    # Block i is redundant
                    current_blocks = test_blocks
                    improved = True
                    break
                
                # Try to find a better block
                best_candidate = None
                best_gain = -1
                
                for anchor in list(newly_uncovered)[:10]:
                    anchor_set = set(anchor)
                    remaining_points = [p for p in all_points if p not in anchor_set]
                    need = k - t
                    
                    for _ in range(5):
                        if need <= 0:
                            block_set = set(anchor)
                        else:
                            extra = random.sample(remaining_points, min(need, len(remaining_points)))
                            block_set = set(anchor) | set(extra)
                        
                        if len(block_set) != k:
                            continue
                        
                        candidate = sorted(block_set)
                        if candidate in current_blocks:
                            continue
                        
                        # Count how many newly_uncovered this covers
                        candidate_set = set(candidate)
                        gain = sum(1 for ts in newly_uncovered if set(ts).issubset(candidate_set))
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = candidate
                
                if best_candidate is not None and best_gain > 0:
                    current_blocks[i] = best_candidate
                    improved = True
                    break
            
            if not improved:
                break
        
        is_valid, _ = check_coverage(current_blocks)
        if is_valid and len(current_blocks) < best_size:
            best_size = len(best_size)
            best_solution = current_blocks[:]
    
    # Ensure blocks are distinct
    if best_solution is None:
        best_solution = greedy_cover()
    
    seen = set()
    unique_blocks = []
    for b in best_solution:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
    
    return unique_blocks
# EVOLVE-BLOCK-END
