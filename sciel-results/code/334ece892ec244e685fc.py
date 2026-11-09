      def batcher_odd_merge(n):
          if n == 1:
              return []
          
          n1 = n // 2
          n2 = n - n1
          
          comps = []
          
          # Recursively sort odd and even
          for a, b in batcher_odd_merge(n1):
              comps.append((2*a + 1, 2*b + 1))
          for a, b in batcher_odd_merge(n2):
              comps.append((2*a, 2*b))
          
          # Merge
          # We define a helper to merge two sorted lists of size n1 and n2
          # which are interleaved starting at index 0.
          
          def merge(n1, n2, offset):
              if n1 == 0:
                  return []
              
              comps = []
              # Layer 1
              for i in range(n1):
                  comps.append((offset + 2*i, offset + 2*i + 1))
              
              # Recursive merge
              # The two sub-problems are:
              # 1. Merging the "lower" parts.
              # 2. Merging the "upper" parts.
              
              # The sizes are ceil((n1+n2)/2) and floor((n1+n2)/2).
              # But the wires are not contiguous.
              
              # Actually, the merge recursion in Batcher's Odd-Merge is:
              # merge(ceil((n1+n2)/2), floor((n1+n2)/2), offset)
              # This is WRONG.
              
              # Correct:
              # The merge splits into two independent merges:
              # 1. On the wires corresponding to the first half of the merged array.
              # 2. On the wires corresponding to the second half.
              
              # For interleaved lists, the "first half" of the merged array
              # is not a contiguous set of wires.
              
              pass
          
          return comps
      