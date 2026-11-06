import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin
    
    best_len = 0
    best_set = set()
    
    # Class to maintain the state of a sequence and its block products
    class SequenceState:
        def __init__(self, nums=None):
            self.nums = sorted(nums) if nums else []
            self.products = set()
            self._recompute()
        
        def _recompute(self):
            self.products = set()
            n = len(self.nums)
            for i in range(n):
                p = 1
                for j in range(i, n):
                    p *= self.nums[j]
                    self.products.add(p)
        
        def try_add(self, x):
            """
            Returns (success, new_products, new_nums)
            If success is False, new_products is None.
            """
            n = len(self.nums)
            # Find insertion index
            k = 0
            while k < n and self.nums[k] < x:
                k += 1
            
            # New blocks are those containing x.
            # Indices in new list: i in [0..k], j in [k..n-1] (relative to new list)
            # In old list:
            # Left part: self.nums[i..k-1]
            # Right part: self.nums[k..j-1]
            
            new_products = set(self.products)
            
            # Check for conflicts with existing products
            # And conflicts among the new products themselves
            
            # We can compute the products of the new blocks efficiently.
            # Let L[i] = product of self.nums[i..k-1] for i in 0..k
            # Let R[j] = product of self.nums[k..j-1] for j in k..n
            # New block product for i, j is L[i] * x * R[j]
            
            # Precompute L
            L = [1] * (k + 1)
            # L[0] = 1 (empty left)
            # L[i] = self.nums[i-1] * L[i-1]? No, L[i] is product of nums[i..k-1].
            # It's easier to compute from k-1 downwards or 0 upwards.
            # L[i] = product of nums[i] to nums[k-1].
            # L[k] = 1.
            # L[i] = nums[i] * L[i+1].
            for i in range(k - 1, -1, -1):
                L[i] = self.nums[i] * L[i + 1]
            
            # Precompute R
            # R[j] = product of nums[k] to nums[j-1].
            # R[k] = 1.
            # R[j+1] = nums[j] * R[j].
            R = [1] * (n - k + 1)
            for j in range(k, n):
                R[j - k + 1] = self.nums[j] * R[j - k]
            
            valid = True
            # Iterate over all new blocks
            # i ranges from 0 to k
            # j ranges from k to n-1
            # The block is nums[i..k-1] + [x] + nums[k..j-1]
            # Product is L[i] * x * R[j-k]
            
            # To optimize, we can just check if the product is in self.products (old)
            # and keep a local set for new products to check against each other.
            # However, since we are just checking validity for a hypothetical add,
            # we can just generate them and check against a temporary set.
            
            temp_new = set()
            for i in range(k + 1):
                left_part = L[i]
                for j in range(k, n):
                    right_part = R[j - k]
                    p = left_part * x * right_part
                    if p in self.products or p in temp_new:
                        valid = False
                        break
                    temp_new.add(p)
                if not valid:
                    break
            
            if valid:
                new_nums = self.nums[:k] + [x] + self.nums[k:]
                return True, new_products, new_nums
            else:
                return False, None, None

    # Initial Greedy Solution
    def greedy_init(skip=None):
        s = set()
        state = SequenceState([])
        for x in range(2, 129):
            if x == skip:
                continue
            if x in s:
                continue
            success, new_products, new_nums = state.try_add(x)
            if success:
                s.add(x)
                state.nums = new_nums
                state.products = new_products
        
        return s, state

    # Run initial greedy
    best_set, best_state = greedy_init()
    best_len = len(best_set)
    
    # Local Search
    # We will perform a loop of:
    # 1. Try to add any valid number.
    # 2. If stuck, try swaps (remove a, add b).
    # 3. If still stuck, restart with a perturbation.
    
    while time.time() < deadline:
        current_set = set(best_set)
        current_state = SequenceState(sorted(current_set))
        
        improved = True
        stuck_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try Additions
            # To make this faster, we can just iterate through candidates.
            # Since N is small, O(N^2) per check is fine.
            valid_adds = []
            for x in range(2, 129):
                if x in current_set:
                    continue
                success, _, _ = current_state.try_add(x)
                if success:
                    valid_adds.append(x)
            
            if valid_adds:
                x = random.choice(valid_adds)
                current_set.add(x)
                current_state.nums, _ = current_state.try_add(x) # Update state
                # Note: try_add returns new_products too, we should update that.
                # Let's re-call or store it.
                # Better: modify try_add to return the state update or just recompute?
                # Recomputing is O(N^2). N is ~60. 3600 ops. Very fast.
                # Let's just recompute for simplicity and safety in state management.
                current_state = SequenceState(sorted(current_set))
                
                improved = True
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = set(current_set)
                    best_state = SequenceState(sorted(best_set))
            else:
                # 2. Try Swaps
                # Find a, b such that S - {a} + {b} is valid.
                # To speed up, we can just pick a random 'a' to remove,
                # then check which 'b' can be added to S-{a}.
                # This is O(N * N * CostOfCheck).
                # CostOfCheck is O(N^2). Total O(N^4) per swap step?
                # N=60. 60^4 = 12,960,000. Might be too slow if done many times.
                
                # Optimization: Only try a few random swaps.
                swaps_found = False
                if len(current_set) > 0:
                    # Pick a random element to remove
                    a = random.choice(sorted(current_set))
                    temp_set = set(current_set)
                    temp_set.remove(a)
                    temp_state = SequenceState(sorted(temp_set))
                    
                    # Now try to add any b not in temp_set
                    for b in range(2, 129):
                        if b in temp_set:
                            continue
                        success, _, _ = temp_state.try_add(b)
                        if success:
                            # Perform the swap
                            current_set = temp_set
                            current_set.add(b)
                            current_state = SequenceState(sorted(current_set))
                            improved = True
                            swaps_found = True
                            
                            if len(current_set) > best_len:
                                best_len = len(current_set)
                                best_set = set(current_set)
                                best_state = SequenceState(sorted(best_set))
                            break # Only do one swap per iteration to keep it fast?
                            # Or continue? Usually one swap is enough to unlock additions.
                    
                if not swaps_found:
                    # Stuck. Break out of inner loop to restart.
                    break
        
        # 3. Restart with perturbation
        # Skip a random number in the greedy pass
        skip = random.randint(2, 128)
        s, state = greedy_init(skip=skip)
        
        if len(s) > best_len:
            best_len = len(s)
            best_set = set(s)
            best_state = state
        
        # Also try to improve the restarted solution a bit
        # (Optional: Run a few local search steps on the restarted solution)
        # To save time, we might just let the next main loop iteration handle it
        # if we update best_set. But best_set is only updated if len > best_len.
        # If len == best_len, we might want to try to improve it.
        # Let's just rely on the main loop.

    return sorted(best_set)
