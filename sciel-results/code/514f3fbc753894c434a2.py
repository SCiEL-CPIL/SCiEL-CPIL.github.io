   def construct(n_rows, k):
       # We want to minimize the max element.
       # We'll use backtracking to find a valid configuration.
       # To improve chances of finding a small scope, we can try to keep the max element low.
       
       # Since the problem is small (7 rows, 6 elements), we can try a DFS that
       # builds rows one by one.
       
       # Optimization: The order of rows doesn't matter for the final set, 
       # but it matters for the search. 
       # We can try to construct rows with "dense" small differences first?
       
       # Let's use a recursive backtracker.
       # State: 
       #   used_diffs: set of integers
       #   rows: list of completed rows
       #   current_row: list of integers for the row currently being built
       
       # To speed up, we can limit the search for the next mark x.
       # If we are looking for a solution with max element <= M, we can bound x.
       # But we don't know M. We can increase M until we find a solution.
       # Binary search on M? Or just incremental?
       # The counting bound is 105. Best known 112.
       # Let's try to find a solution with M = 105, 106, ...
       
       # However, a single DFS run might be slow if it explores bad branches.
       # Randomized Restart DFS is a good heuristic.
       
       def is_valid_extension(row, x, used_diffs):
           new_diffs = {x - a for a in row}
           if new_diffs & used_diffs:
               return False
           # Check internal uniqueness of new_diffs? 
           # Since x is new and larger than all a in row, x-a are distinct.
           # But we must also ensure new_diffs don't conflict with existing diffs in 'row'?
           # 'row' is a Golomb ruler by construction if we checked at each step.
           # So we just need new_diffs disjoint from used_diffs.
           # Wait, used_diffs contains diffs from PREVIOUS rows.
           # What about diffs within the CURRENT row?
           # If we add x, the new diffs are x-a.
           # The old diffs in 'row' are already in 'used_diffs'? NO.
           # We should maintain 'current_row_diffs' separately or update 'used_diffs' incrementally?
           # If we update 'used_diffs' incrementally, we lose the ability to backtrack easily unless we copy the set.
           # Copying a set of size ~100 is cheap.
           
           # Actually, it's safer to keep 'used_diffs' as the union of all previous rows.
           # And check if new_diffs intersects used_diffs.
           # We also need to ensure that the new row itself is a Golomb ruler.
           # If we build the row step-by-step and ensure that each new x creates diffs
           # that are distinct from each other (trivial) and distinct from previous diffs IN THE SAME ROW,
           # then the row is a Golomb ruler.
           # So we need to track diffs within the current row too.
           
           # Let's pass 'row_diffs' as well.
           pass

       # Revised State:
       # used_global: set of diffs from completed rows
       # row: current row
       # row_diffs: set of diffs within current row
       
       def dfs(row_idx, used_global, current_row, row_diffs, max_limit):
           if row_idx == n_rows:
               return True, current_row # Return success
           
           if len(current_row) == k + 1:
               # Row complete
               new_used_global = used_global | row_diffs
               # Recurse to next row
               # Start next row with [0]
               res, _ = dfs(row_idx + 1, new_used_global, [0], set(), max_limit)
               if res:
                   return True, None
               return False, None
           
           # Try to add next mark
           last = current_row[-1]
           # Start from last + 1 up to max_limit
           # Heuristic: try smaller values first
           for x in range(last + 1, max_limit + 1):
               new_diffs = {x - a for a in current_row}
               # Check conflicts with global
               if new_diffs & used_global:
                   continue
               # Check conflicts with current row diffs
               if new_diffs & row_diffs:
                   continue
               
               # Valid extension
               new_current_row = current_row + [x]
               new_row_diffs = row_diffs | new_diffs
               
               res, _ = dfs(row_idx, used_global, new_current_row, new_row_diffs, max_limit)
               if res:
                   return True, None
           
           return False, None

       # We need to store the final rows.
       # The above DFS doesn't easily return the full set of rows if we just return True.
       # We can modify it to build the list of rows.
       
       # Also, we need to find the minimal max_limit.
       # We can try max_limit = 105, 106, ...
       # If DFS is fast, this is fine.
       
       # To make DFS faster, we can add symmetry breaking or better ordering.
       # For the first row, we can fix it to a specific Golomb ruler?
       # No, that might be suboptimal.
       
       # Let's implement the DFS with row collection.
       
       best_rows = None
       
       def search(used_global, rows, current_row, row_diffs, max_limit):
           nonlocal best_rows
           
           if len(rows) == n_rows:
               best_rows = rows
               return True
           
           if len(current_row) == k + 1:
               # Finish current row
               new_rows = rows + [current_row]
               new_used_global = used_global | row_diffs
               # Start next row
               return search(new_used_global, new_rows, [0], set(), max_limit)
           
           last = current_row[-1]
           # Heuristic: try x in a specific range
           # We can also randomize the order of x to help with restarts
           # But for minimal scope, we want small x.
           
           for x in range(last + 1, max_limit + 1):
               new_diffs = {x - a for a in current_row}
               if new_diffs & used_global:
                   continue
               if new_diffs & row_diffs:
                   continue
               
               if search(used_global, rows, current_row + [x], row_diffs | new_diffs, max_limit):
                   return True
           
           return False

       # Try increasing max_limit
       # Counting bound: n_rows * (k*(k+1)//2)
       lower_bound = n_rows * (k * (k + 1) // 2)
       
       # The seed program is greedy. It likely produces a large scope.
       # We can start from lower_bound and go up.
       # If we don't find a solution in reasonable time for a given max_limit, we increase it.
       # But DFS might hang. We need a timeout or node limit.
       
       # Given 120s, and small N,K, we can try a few values.
       
       # To improve performance, we can use a "fail-first" heuristic?
       # Or just rely on the small size.
       
       # Let's add a node counter to prevent infinite loops/hangs.
       nodes_visited = 0
       limit_nodes = 1000000 # Adjust based on speed
       
       def search_limited(used_global, rows, current_row, row_diffs, max_limit):
           nonlocal best_rows, nodes_visited
           
           if len(rows) == n_rows:
               best_rows = rows
               return True
           
           if len(current_row) == k + 1:
               new_rows = rows + [current_row]
               new_used_global = used_global | row_diffs
               return search_limited(new_used_global, new_rows, [0], set(), max_limit)
           
           last = current_row[-1]
           
           # Optimization: If the number of available differences is too low, prune?
           # Hard to check easily.
           
           for x in range(last + 1, max_limit + 1):
               nodes_visited += 1
               if nodes_visited > limit_nodes:
                   return False
               
               new_diffs = {x - a for a in current_row}
               if new_diffs & used_global:
                   continue
               if new_diffs & row_diffs:
                   continue
               
               if search_limited(used_global, rows, current_row + [x], row_diffs | new_diffs, max_limit):
                   return True
           
           return False

       # Try to find a solution
       # Start from lower_bound
       # If it fails, increment max_limit
       
       # Note: The problem is symmetric. The order of rows doesn't matter.
       # We can enforce an ordering on the rows to reduce symmetry?
       # e.g., the max element of row i < max element of row i+1?
       # This might help.
       
       # Let's try without symmetry breaking first, but with a reasonable node limit.
       
       # We can also try randomizing the start of the search or the order of x.
       # But for minimization, small x first is best.
       
       # Let's run the search for max_limit from lower_bound upwards.
       
       # To avoid hanging on unsolvable small limits, we use the node limit.
       
       # If we find a solution, we return it.
       
       # If we fail to find any solution within a reasonable max_limit (e.g., 200),
       # we fall back to the greedy method or return the best partial?
       # The task requires a valid result.
       
       # Let's assume a solution exists for max_limit up to some reasonable number.
       
       # One more optimization: 
       # We can precompute valid Golomb rulers of length k+1 with max element <= max_limit.
       # Then the problem becomes: choose n_rows rulers from this list such that their difference sets are disjoint.
       # This is a Set Packing problem.
       # For k=5, the number of such rulers is small.
       # We can generate all candidates and then use backtracking to select disjoint ones.
       # This separates the generation of valid rows from the selection.
       
       # Step 1: Generate all Golomb rulers of length k+1 with max element <= max_limit.
       # Step 2: Use backtracking to pick n_rows disjoint rulers.
       
       # This is likely faster because the inner loop (checking diffs) is replaced by set intersection of precomputed sets.
       
       def generate_rulers(max_limit):
           rulers = []
           # DFS to generate one ruler
           def gen(row, row_diffs):
               if len(row) == k + 1:
                   rulers.append((row, row_diffs))
                   return
               last = row[-1]
               for x in range(last + 1, max_limit + 1):
                   new_diffs = {x - a for a in row}
                   if new_diffs & row_diffs:
                       continue
                   gen(row + [x], row_diffs | new_diffs)
           
           gen([0], set())
           return rulers

       # Step 2: Select disjoint rulers
       def select_rulers(rulers, n_needed, used_diffs, selected):
           if len(selected) == n_needed:
               return selected
           
           # Heuristic: try rulers with smaller max element first?
           # Or just iterate.
           
           for i, (row, diffs) in enumerate(rulers):
               if diffs & used_diffs:
                   continue
               
               # Symmetry breaking: 
               # We can enforce that the index of the selected ruler increases?
               # Or that the max element is non-decreasing?
               # Let's enforce max element non-decreasing to break symmetry of row order.
               if selected:
                   if row[-1] < selected[-1][-1]:
                       continue
               
               new_selected = selected + [row]
               new_used = used_diffs | diffs
               
               res = select_rulers(rulers, n_needed, new_used, new_selected)
               if res:
                   return res
           
           return None

       # Combine:
       # For max_limit in range(lower_bound, 200):
       #    rulers = generate_rulers(max_limit)
       #    if len(rulers) < n_rows: continue
       #    res = select_rulers(rulers, n_rows, set(), [])
       #    if res: return res
       
       # This approach is clean.
       # Generating rulers for max_limit=105 might take a bit, but k=5 is small.
       # Number of Golomb rulers of order 6 is known to be small (hundreds?).
       
       # Let's implement this.
       
       # One issue: `generate_rulers` might be slow if max_limit is large.
       # But we start from the lower bound.
       
       # To speed up `select_rulers`, we can sort rulers by their max element.
       
       # Let's code this.

       lower_bound = n_rows * (k * (k + 1) // 2)
       
       # If lower_bound is too low, we might waste time.
       # But it's a hard bound.
       
       # We'll try max_limit from lower_bound to lower_bound + 50.
       # If not found, we fall back to greedy?
       # The greedy is guaranteed to work (though suboptimal).
       # But we want to beat the seed.
       
       # Let's add a timeout to the whole construct function.
       
       import time
       start_time = time.time()
       
       for M in range(lower_bound, lower_bound + 100):
           if time.time() - start_time > 100:
               break
           
           # Generate rulers with max element <= M
           # Optimization: only generate if we can fit n_rows disjoint ones?
           # Just generate.
           
           rulers = []
           def gen(row, row_diffs):
               if len(row) == k + 1:
                   rulers.append((row, row_diffs))
                   return
               last = row[-1]
               # Pruning: if even with max difference M, we can't fit remaining marks?
               # Not easy.
               for x in range(last + 1, M + 1):
                   new_diffs = {x - a for a in row}
                   if new_diffs & row_diffs:
                       continue
                   gen(row + [x], row_diffs | new_diffs)
           
           gen([0], set())
           
           if len(rulers) < n_rows:
               continue
           
           # Sort rulers by max element to help symmetry breaking
           rulers.sort(key=lambda r: r[0][-1])
           
           # Backtrack to select
           def select(used_diffs, selected, start_idx):
               if len(selected) == n_rows:
                   return selected
               
               for i in range(start_idx, len(rulers)):
                   row, diffs = rulers[i]
                   if diffs & used_diffs:
                       continue
                   
                   # Symmetry breaking: ensure non-decreasing max element
                   # Since sorted, row[-1] >= selected[-1][-1] is always true if we pick i > prev_idx?
                   # Not necessarily, but if we enforce i > prev_idx, we avoid permutations of the same set.
                   
                   new_selected = selected + [row]
                   new_used = used_diffs | diffs
                   
                   res = select(new_used, new_selected, i + 1)
                   if res:
                       return res
               
               return None
           
           result = select(set(), [], 0)
           if result:
               return result
       
       # Fallback to greedy if no solution found in time/range
       used, rows = set(), []
       for _ in range(n_rows):
           row, x = [0], 0
           while len(row) < k + 1:
               x += 1
               new = {x - a for a in row}
               if not new & used:
                   row.append(x)
                   used |= new
           rows.append(row)
       return rows

   # The code above is inside construct.
   # We need to make sure it handles general n_rows, k.
   # The lower bound calculation is general.
   # The generation and selection are general.
   
   # One potential issue: recursion depth.
   # k is small (5), so depth is 5.
   # n_rows is small (7).
   # So recursion is safe.
   
   # Performance:
   # Generating rulers for M=105.
   # Number of nodes in gen tree?
   # Depth 5. Branching factor?
   # At step 1, x in 1..105.
   # At step 2, x in x1+1..105.
   # It might be large.
   # But we prune with `new_diffs & row_diffs`.
   # For Golomb rulers, this prunes a lot.
   
   # To be safe, we can add a node limit to gen as well.
   
   # Let's refine the gen function with a limit.

