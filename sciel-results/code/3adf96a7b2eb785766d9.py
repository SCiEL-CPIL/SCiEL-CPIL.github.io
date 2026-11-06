   def construct():
       N = 18
       comparators = []
       
       def merge(A_start, A_len, B_start, B_len):
           # Merges two sorted sequences located at wires [A_start, A_start+A_len) and [B_start, B_start+B_len)
           # into a single sorted sequence at [A_start, A_start+A_len+B_len)
           # Note: This requires the output to be in the combined range.
           # Standard Batcher merge assumes the two lists are adjacent in memory?
           # Or does it operate in-place on the combined range?
           # Usually, we generate comparators that operate on the specific wire indices.
           
           if A_len == 0:
               return
           if B_len == 0:
               return
           
           # Split A into even and odd
           # A_even indices: A_start, A_start+2, ...
           # A_odd indices: A_start+1, A_start+3, ...
           A_even_len = (A_len + 1) // 2
           A_odd_len = A_len // 2
           
           B_even_len = (B_len + 1) // 2
           B_odd_len = B_len // 2
           
           # The "Even" part of the merge result comes from merging A_even and B_even
           # The "Odd" part comes from merging A_odd and B_odd
           
           # However, the indices in the output are interleaved.
           # Output index 0 (Even) is from A_even/B_even merge.
           # Output index 1 (Odd) is from A_odd/B_odd merge.
           
           # This recursive definition is tricky because the "wires" for the sub-merges are not contiguous.
           # A_even is on wires A_start, A_start+2...
           # B_even is on wires B_start, B_start+2...
           
           # We need to map the logical sub-array indices to physical wire indices.
           
           # Let's define a helper that takes a list of wire indices.
           
           def merge_wires(wires_a, wires_b):
               # wires_a and wires_b are lists of wire indices, each representing a sorted sequence.
               # We want to produce comparators that merge them into a single sorted sequence
               # on the combined list of wires (sorted by index? No, the output order is determined by the merge logic).
               
               len_a = len(wires_a)
               len_b = len(wires_b)
               
               if len_a == 0: return
               if len_b == 0: return
               
               if len_a == 1 and len_b == 1:
                   # Compare the two
                   i, j = wires_a[0], wires_b[0]
                   if i > j: i, j = j, i
                   comparators.append((i, j))
                   return
               
               # Split into even and odd logical positions
               a_even = wires_a[0::2]
               a_odd = wires_a[1::2]
               b_even = wires_b[0::2]
               b_odd = wires_b[1::2]
               
               # Recursively merge evens and odds
               merge_wires(a_even, b_even)
               merge_wires(a_odd, b_odd)
               
               # Now perform the "interleaving" comparisons
               # The output sequence is:
               # out[0] from (a_even, b_even) merge result at pos 0
               # out[1] from (a_odd, b_odd) merge result at pos 0
               # out[2] from (a_even, b_even) merge result at pos 1
               # out[3] from (a_odd, b_odd) merge result at pos 1
               # ...
               
               # The comparators needed to fix the order between out[2k-1] and out[2k]
               # and out[2k] and out[2k+1]?
               
               # Standard Odd-Even Merge comparators after recursive merges:
               # Compare (out[2i-1], out[2i]) for i=1..
               # and (out[2i], out[2i+1])? No.
               
               # The specific comparators are:
               # For i in range(min(len(a_even), len(b_even)) - 1):
               #    Compare(out[2i+1], out[2i+2]) ?
               
               # Let's look at the standard formula.
               # After merging evens and odds, we have:
               # E = merge(A_even, B_even)
               # O = merge(A_odd, B_odd)
               # The final sequence is E[0], O[0], E[1], O[1], ...
               # We need to compare:
               # 1. O[i] and E[i+1] for i = 0 to min(len(O), len(E)-1) - 1
               #    These are at positions 2i+1 and 2i+2.
               
               # How to map E and O back to wire indices?
               # The wires for E are the union of a_even and b_even wires, but sorted?
               # No, the merge_wires function returns nothing, it just adds comparators.
               # But we need to know WHICH wire holds E[0], E[1], etc.
               
               # This approach fails because `merge_wires` doesn't track the output positions.
               
               # We need a function that returns the list of wire indices in the order they appear in the sorted output.
               
               def merge_return(wires_a, wires_b):
                   # Returns list of wire indices in sorted output order
                   len_a = len(wires_a)
                   len_b = len(wires_b)
                   
                   if len_a == 0: return wires_b
                   if len_b == 0: return wires_a
                   
                   if len_a == 1 and len_b == 1:
                       i, j = wires_a[0], wires_b[0]
                       if i > j: i, j = j, i
                       comparators.append((i, j))
                       return [i, j]
                   
                   a_even = wires_a[0::2]
                   a_odd = wires_a[1::2]
                   b_even = wires_b[0::2]
                   b_odd = wires_b[1::2]
                   
                   e_res = merge_return(a_even, b_even)
                   o_res = merge_return(a_odd, b_odd)
                   
                   # Now interleave e_res and o_res
                   # e_res[0], o_res[0], e_res[1], o_res[1]...
                   result = []
                   min_len = min(len(e_res), len(o_res))
                   for i in range(min_len):
                       result.append(e_res[i])
                       result.append(o_res[i])
                       # Compare o_res[i] and e_res[i+1]
                       # o_res[i] is at result[-1]
                       # e_res[i+1] is at result[-1] + 1? No, it's the next even.
                       # Wait, the comparator is between the last odd and the next even.
                       # Position 2i+1 (o_res[i]) and 2i+2 (e_res[i+1])
                       if i + 1 < len(e_res):
                           w1 = o_res[i]
                           w2 = e_res[i+1]
                           if w1 > w2: w1, w2 = w2, w1
                           comparators.append((w1, w2))
                   
                   if len(e_res) > min_len:
                       result.extend(e_res[min_len:])
                   if len(o_res) > min_len:
                       result.extend(o_res[min_len:])
                       
                   return result

               # Initialize with two halves
               # For sorting N, we split into two halves, sort them, then merge.
               # But we need to sort the halves first.
               # This recursive structure implies we need a `sort` function.
               
               def sort(wires):
                   n = len(wires)
                   if n <= 1: return wires
                   
                   mid = n // 2
                   left = wires[:mid]
                   right = wires[mid:]
                   
                   # Sort left and right
                   left_sorted = sort(left)
                   right_sorted = sort(right)
                   
                   # Merge them
                   return merge_return(left_sorted, right_sorted)
               
               # Start with all wires
               all_wires = list(range(N))
               sort(all_wires)
               
               return comparators

       return construct()
   