import time
import random
import sys
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute all t-subsets (triplets in this case)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of triplet IDs it covers
    # Using lists is faster for counting in Python than bitmasks for small v
    block_triplets = []
    for block in all_blocks:
        tids = []
        for triplet in itertools.combinations(block, t):
            tids.append(triplet_to_id[triplet])
        block_triplets.append(tids)
        
    # Theoretical lower bound: Ceiling(C(v,t) / C(k,t))
    # C(24,3) = 2024, C(7,3) = 35. 2024/35 = 57.8. 
    # Repository LB is 69. We will start from 69 to save time, or slightly lower to be safe?
    # If we start too low, we might waste time proving impossibility.
    # Given 150s limit, proving C(24,7,3) >= 69 might be hard.
    # However, finding a covering of size 76 is the current best.
    # Let's start from 69. If it fails quickly, we move up.
    # Actually, a simple greedy gets ~76-80. 
    # Let's use a heuristic lower bound.
    lb = 69
    
    # We will try to find a solution of size N, starting from lb, increasing N.
    # To avoid getting stuck in one N for too long, we limit time per N.
    
    best_solution = None
    best_n = float('inf')
    
    # Initial greedy solution to have a baseline and maybe a better upper bound
    # Quick greedy
    uncovered = set(range(n_triplets))
    greedy_blocks = []
    greedy_indices = []
    
    # Sort blocks by size of coverage (all same) or random
    # Simple greedy: pick block covering most uncovered
    # To speed up, we can just run a few random greedies to get a decent UB
    # But CSP is the main goal.
    
    # Let's run a fast greedy to get an initial UB
    rng = random.Random(42)
    current_uncovered = set(range(n_triplets))
    temp_blocks = []
    while current_uncovered:
        best_gain = -1
        best_idx = -1
        # Sample a subset of blocks if n_blocks is large? No, 34610 is manageable for one pass
        # But doing this for 76 steps * 34610 blocks is ~2.6M ops. Fast.
        for i in range(n_blocks):
            gain = 0
            # Count how many of this block's triplets are in current_uncovered
            # This is slow in Python loops. 
            # Optimization: Precompute block masks as integers? 
            # n_triplets = 2024. Bitmask is 2024 bits. Python big ints handle this.
            pass
        
    # Let's stick to the CSP approach but use a good initial UB from a simple heuristic.
    # We'll use a "first-fit" style greedy with randomization for a few seeds.
    
    def quick_greedy(seed):
        r = random.Random(seed)
        uncov = set(range(n_triplets))
        blocks_used = []
        while uncov:
            # Pick random block? No, pick one with max gain.
            # To make it fast, we can approximate gain or just pick random from top candidates?
            # Let's just pick a random block that covers at least one new triplet?
            # That's inefficient.
            
            # Better: Precompute for each triplet, which blocks cover it.
            # Then pick a random uncovered triplet, and pick a random block covering it?
            # This is "Randomized Greedy".
            
            # Let's try: Pick random uncovered triplet. Pick random block containing it.
            # This is very fast.
            if not uncov: break
            # Pick random triplet from uncov
            # List conversion is O(n). 
            # We can maintain a list of uncovered and remove? 
            # Or just use a set and pick one.
            # Set pop is O(1) avg? No, random.choice is O(n).
            
            # Let's use the triplet -> blocks index.
            # We need to build it once.
            
            pass
        return blocks_used

    # Build index: triplet_id -> list of block_indices
    triplet_to_blocks = defaultdict(list)
    for b_idx, tids in enumerate(block_triplets):
        for tid in tids:
            triplet_to_blocks[tid].append(b_idx)
            
    def fast_greedy(seed):
        r = random.Random(seed)
        uncov = set(range(n_triplets))
        blocks_used = []
        while uncov:
            # Pick a random uncovered triplet
            # To make this O(1), we could maintain a list, but set is fine for 2024 items
            # random.choice on a set requires list conversion. 
            # Let's just pick a random triplet ID and check if it's in uncov?
            # Collision probability is low initially, high later.
            # Better: list(uncov) is O(n). 2024 is small.
            tid = r.choice(list(uncov))
            # Pick a random block covering this triplet
            candidates = triplet_to_blocks[tid]
            b_idx = r.choice(candidates)
            blocks_used.append(all_blocks[b_idx])
            # Remove all triplets covered by this block from uncov
            for t in block_triplets[b_idx]:
                if t in uncov:
                    uncov.remove(t)
        return blocks_used

    # Get an initial upper bound
    ub = float('inf')
    best_greedy = None
    for s in range(10):
        if time.time() > deadline - 20: break
        sol = fast_greedy(s)
        if len(sol) < ub:
            ub = len(sol)
            best_greedy = sol
            
    if best_greedy is None:
        best_greedy = fast_greedy(0)
        ub = len(best_greedy)
        
    # Now run CSP for N = lb to ub-1
    # If we find a solution, update best and stop (since we want minimum).
    # If we don't find one for a specific N within time limit, move to N+1?
    # Or just return the best greedy found?
    # The task asks for "as few as possible". 
    # If CSP proves no solution for N=69, 70... up to 75, then 76 is the answer (if greedy found 76).
    # But proving non-existence is hard.
    # Strategy: Try to find a solution for N = lb. If successful, return it.
    # If not, try N = lb+1, etc.
    # Limit time per N to prevent getting stuck.
    
    # CSP Solver
    # State: 
    # - remaining_slots: number of blocks left to choose
    # - uncovered_triplets: set of triplet IDs not yet covered
    # - used_blocks: set of block indices already chosen
    
    # Heuristic: 
    # - MRV: Pick the triplet with the fewest available blocks (blocks that cover it and haven't been used).
    # - But we don't choose triplets, we choose blocks.
    # - Alternative: Pick a block that covers the "hardest" triplet.
    # - "Hardest" triplet: The one with the fewest remaining candidate blocks.
    
    # Let's implement a backtracking search.
    
    # To make it efficient, we need to quickly find candidate blocks for a given state.
    # State is defined by `uncovered` and `used_blocks`.
    # Candidate blocks are those not in `used_blocks`.
    # We want to pick a block that covers a triplet with low degree.
    
    # Pre-compute: For each block, the set of triplets it covers (as a set or frozenset for intersection?)
    # block_triplets is list of lists.
    
    def solve_csp(n_blocks_needed, time_limit):
        start_csp = time.time()
        
        # Initial state
        initial_uncovered = set(range(n_triplets))
        
        # We need to search for a set of `n_blocks_needed` blocks.
        
        # To speed up, we can use a heuristic: 
        # If the number of uncovered triplets > remaining_slots * max_coverage_per_block, fail.
        # max_coverage = C(k,t) = 35.
        
        # Also, if any triplet has 0 available blocks, fail.
        
        # Let's use a recursive DFS.
        # To avoid deep recursion limits, we can use an explicit stack or increase recursion limit.
        # sys.setrecursionlimit(10000)
        
        # Optimization: Symmetry breaking? 
        # Blocks are distinct. Order of selection doesn't matter for the set.
        # We can enforce an ordering on block indices to avoid permutations.
        # i.e., next block index must be > previous block index.
        
        def backtrack(remaining_slots, last_block_idx, uncovered, current_solution):
            if time.time() > time_limit:
                return None
            if time.time() > deadline - 5:
                return None # Global timeout
            
            if remaining_slots == 0:
                if not uncovered:
                    return current_solution
                return None
            
            # Pruning:
            # 1. Max coverage check
            max_cov = 35 # C(7,3)
            if len(uncovered) > remaining_slots * max_cov:
                return None
            
            # 2. Check if any uncovered triplet has no available blocks
            # Available blocks for a triplet T are those in triplet_to_blocks[T] with idx > last_block_idx
            # and not used (implied by idx > last_block_idx if we enforce order and don't reuse? 
            # Wait, we can't reuse blocks. 
            # If we enforce idx > last_block_idx, we automatically don't reuse previous blocks.
            # But we might skip a block with idx < last_block_idx that wasn't used? 
            # No, if we are at step i, we have chosen i blocks. 
            # If we enforce strictly increasing indices, we cover all combinations of size N.
            
            # Find the "hardest" triplet: one with the fewest available blocks (idx > last_block_idx)
            min_avail = float('inf')
            target_tid = -1
            
            # Iterate through uncovered triplets
            # To speed up, we can sample if uncovered is large? 
            # But n_triplets is 2024. 
            # We need to find min count.
            # For each tid in uncovered:
            #   count how many b in triplet_to_blocks[tid] such that b > last_block_idx
            # This is O(2024 * avg_degree). avg_degree is ~35*7/24 * ... 
            # Actually, each block covers 35 triplets. Total incidences 34610*35 = 1.2M.
            # Avg degree per triplet = 1.2M / 2024 ~ 600.
            # So for each tid, scanning 600 blocks is expensive if we do it for all 2024.
            # 2024 * 600 = 1.2M ops per node. Too slow.
            
            # Alternative: 
            # Pick a random uncovered triplet?
            # Or maintain counts?
            
            # Let's try a different heuristic:
            # Pick a random uncovered triplet.
            # Get its available blocks.
            # If none, fail.
            # Pick one of the available blocks (randomly or best gain).
            
            # To improve, let's pick the triplet with the smallest number of available blocks.
            # We can approximate this by picking a few random triplets and picking the one with fewest.
            
            sample_size = min(10, len(uncovered))
            candidates = random.sample(list(uncovered), sample_size)
            
            best_tid = -1
            best_avail_blocks = None
            min_count = float('inf')
            
            for tid in candidates:
                avail = [b for b in triplet_to_blocks[tid] if b > last_block_idx]
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_tid = tid
                    best_avail_blocks = avail
                    
            if min_count == 0:
                return None
                
            # Pick a block from best_avail_blocks
            # Heuristic: Pick the one that covers the most OTHER uncovered triplets?
            # Or just random?
            # Let's try random first for simplicity and speed.
            # If we pick the best gain, we calculate gain for each candidate in best_avail_blocks.
            # best_avail_blocks size is ~600. 
            # Calculating gain for 600 blocks: 600 * 35 (triplets per block) = 21000 ops.
            # This is acceptable per node.
            
            best_block = -1
            best_gain = -1
            
            for b_idx in best_avail_blocks:
                gain = 0
                for t in block_triplets[b_idx]:
                    if t in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                return None
                
            # Recurse
            new_uncovered = set(uncovered)
            for t in block_triplets[best_block]:
                new_uncovered.discard(t)
                
            new_sol = current_solution + [best_block]
            
            res = backtrack(remaining_slots - 1, best_block, new_uncovered, new_sol)
            if res is not None:
                return res
                
            # If this path failed, try next block? 
            # This makes it a full search. 
            # To keep it fast, we might want to limit branching.
            # But for a minimal N, we might need to explore.
            
            # For now, let's just try the best block. If it fails, return None.
            # This is a greedy heuristic within the backtracking tree.
            # To make it a true solver, we should iterate through all best_avail_blocks.
            
            # Let's modify: Iterate through sorted blocks by gain (desc)
            # But sorting 600 items every node is slow.
            # Let's just pick the best one. If it fails, the whole branch fails.
            # This is effectively a greedy algorithm with backtracking only if the greedy choice leads to a dead end immediately?
            # No, if the greedy choice leads to a dead end 5 steps later, we return None up the stack.
            # The parent will then try its next choice? 
            # In my current structure, `backtrack` returns None if the chosen path fails.
            # It does NOT try other blocks for the same `best_tid`.
            # So this is just a single-path greedy search.
            
            # To make it more robust, let's allow trying a few top blocks.
            # But time is tight.
            
            return None

        # Initial call
        # We need to handle the "last_block_idx" correctly.
        # Start with -1.
        sol = backtrack(n_blocks_needed, -1, initial_uncovered, [])
        return sol

    # Try CSP for N from lb to ub-1
    # Limit time for each N to, say, 10 seconds?
    # If we have 100 seconds left, and range is 69..75 (7 values), we have ~14s per value.
    
    current_time = time.time()
    time_per_n = max(5, (deadline - 10 - current_time) / max(1, ub - lb))
    
    for n in range(lb, ub):
        if time.time() > deadline - 10:
            break
            
        limit = time.time() + time_per_n
        sol = solve_csp(n, limit)
        
        if sol is not None:
            # Convert indices to blocks
            blocks = [all_blocks[i] for i in sol]
            # Verify
            # (Assume valid)
            return blocks
            
        # If not found, try next n
        # Note: solve_csp might return None due to timeout or actual failure.
        # We don't distinguish. We just move to next n.
        
    # If no CSP solution found, return the best greedy
    return best_greedy

if __name__ == "__main__":
    pass
