import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
        n = len(remaining_weights)
        if n == 0:
            return -1
        
        # Scale to integers to avoid float precision issues in DP
        # Assuming weights are reasonable (e.g., < 1000)
        scale = 1000
        C = int(remaining_capacity * scale)
        
        # Filter out items that are too heavy
        valid_indices = []
        weights_int = []
        values = []
        for i in range(n):
            w = int(remaining_weights[i] * scale)
            if w <= C and w > 0:
                valid_indices.append(i)
                weights_int.append(w)
                values.append(remaining_values[i])
        
        if not valid_indices:
            return -1
            
        # If no items fit, return -1
        # Also if max value is 0, return -1
        if max(values) <= 0:
            return -1

        # 0/1 Knapsack DP
        # dp[c] = max value for capacity c
        # We need to track which item was taken to reconstruct.
        # Using a 2D array for reconstruction is O(N*C) space.
        # Or we can store `choice[i][c]` = 1 if item i taken for capacity c.
        
        # Optimization: If N*C is small, 2D is fine.
        # N <= 100, C <= 12500. 1.25M bools.
        
        # Let's use a list of lists for `take`
        # take[i][c] indicates if item i is included in the optimal solution for capacity c
        # This is memory heavy but okay.
        
        # Alternative: Just find the best item greedily using Lagrangian?
        # No, let's stick to DP for optimality.
        
        # To save memory/time, we can just compute the DP and then backtrack.
        # Backtracking requires knowing the state.
        
        # Let's use a 2D DP table for values, and a parallel table for parents.
        # parent[i][c] = index of the item that was last added to reach state (i, c)?
        # No, standard backtracking:
        # if dp[i][c] == dp[i-1][c]: item i not taken.
        # else: item i taken.
        
        # We only need to find ONE item that is part of an optimal solution.
        # We can compute dp table.
        # Then check each item i (in order of preference, e.g., high value) to see if it can be part of an optimal solution.
        # Condition for item i to be in an optimal solution for capacity C:
        # values[i] + dp_excluding_i[C - w_i] == dp_all[C]
        # Computing dp_excluding_i for all i is O(N^2 C). Too slow.
        
        # Better: Backtrack the single optimal path.
        # We compute dp[i][c] for all i, c.
        # Then start at i=n, c=C.
        # While i > 0:
        #   if dp[i][c] == dp[i-1][c]:
        #      i -= 1
        #   else:
        #      # Item i-1 is taken
        #      return valid_indices[i-1]
        #      c -= weights_int[i-1]
        #      i -= 1
        #      # We can return immediately upon finding the first taken item?
        #      # Yes, any item in the optimal set is a valid "next item".
        
        # This backtracking is O(N).
        # The DP calculation is O(N*C).
        
        # Let's implement this.
        
        # Memory check: 100 * 12500 floats = 1.25M * 8 bytes = 10MB. OK.
        
        dp = [[0] * (C + 1) for _ in range(n + 1)]
        
        # Fill DP table
        # We only need to consider valid items.
        # But the indices in `valid_indices` map to original indices.
        # It's easier to run DP on the filtered list and map back.
        
        m = len(valid_indices)
        if m == 0:
            return -1
            
        # dp[i][c] for first i items in valid list
        dp = [[0] * (C + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            w = weights_int[i-1]
            v = values[i-1]
            prev_dp = dp[i-1]
            curr_dp = dp[i]
            # Optimization: slice assignment or loop
            # curr_dp[c] = max(prev_dp[c], prev_dp[c-w] + v)
            # For c < w, curr_dp[c] = prev_dp[c]
            # For c >= w, curr_dp[c] = max(...)
            
            # Copy prev to curr first (case item not taken)
            curr_dp[:] = prev_dp[:]
            
            # Update for item taken
            # Loop from w to C
            for c in range(w, C + 1):
                val = prev_dp[c - w] + v
                if val > curr_dp[c]:
                    curr_dp[c] = val

        # Backtrack to find one item
        c = C
        for i in range(m, 0, -1):
            if dp[i][c] != dp[i-1][c]:
                # Item i-1 was taken
                return valid_indices[i-1]
            else:
                # Item i-1 not taken
                pass
            # Note: if taken, we should reduce c, but since we return immediately, we don't need to.
            # Wait, if we return immediately, we just need to know IF it was taken.
            # The condition `dp[i][c] != dp[i-1][c]` confirms item i-1 is part of *an* optimal solution for capacity c.
            # Is it possible that `dp[i][c] == dp[i-1][c]` but item i-1 *could* have been taken?
            # Yes, if there are ties. But we just need *one* valid item.
            # If `dp[i][c] > dp[i-1][c]`, it MUST be taken.
            # If `dp[i][c] == dp[i-1][c]`, it might or might not be.
            # However, if we skip it, we are still on an optimal path for the remaining items.
            # So we can just return the first item that is *definitely* required or *chosen* in this specific DP path.
            # The standard backtracking logic:
            # if dp[i][c] == dp[i-1][c]: item not taken in this specific optimal solution representation.
            # else: item taken.
            
            # So we just return the first one found where `dp[i][c] > dp[i-1][c]`?
            # Actually, if `dp[i][c] == dp[i-1][c]`, we move to i-1 with same c.
            # If `dp[i][c] > dp[i-1][c]`, we found an item.
            
            # Wait, what if `dp[i][c] == dp[i-1][c]` but item i-1 *could* be part of the solution?
            # It doesn't matter. We just need to return *a* valid item.
            # If we skip it, we eventually find another one or reach 0.
            # If we reach 0 without finding one, it means value is 0?
            # If max value > 0, there must be at least one item taken.
            
        return -1
