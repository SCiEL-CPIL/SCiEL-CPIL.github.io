   def construct():
       import time
       start = time.time()
       limit = start + 145
       
       x = np.ones(N, dtype=np.int8)
       A = np.convolve(x, x, mode='full')
       S = np.sum(x)
       
       def get_max_side(A):
           # A length 2N-1. Center N-1.
           # We want max abs of A excluding center.
           # Slice: A[0:N-1] and A[N:2N-1]
           return np.max(np.abs(np.concatenate((A[:N-1], A[N:]))))
       
       current_max = get_max_side(A)
       current_score = current_max / (S*S)
       
       # Local search
       while time.time() < limit:
           improved = False
           for i in range(N):
               # Flip x[i]
               old_val = x[i]
               x[i] = -old_val
               
               # Update S
               S_new = S - 2*old_val
               
               # Update A
               # A_k = sum x_n x_{n+k}
               # Change in A_k is -2 * x_i * x_{i+k}
               # Indices for k:
               # k = j - i
               # A is indexed by k + (N-1)
               # We need to update A[k] for all k where x_i is involved.
               # x_i is at index i.
               # Pairs (i, i+k) -> k = j - i.
               # j ranges 0..N-1.
               # So k ranges -i .. N-1-i.
               
               # Vectorized update of A
               # The term to subtract/add is 2 * old_val * x[j]
               # A[k] -= 2 * old_val * x[j]
               # k = j - i
               # So A[j-i] -= 2 * old_val * x[j]
               
               # Note: A is stored with offset.
               # A_array[idx] corresponds to k = idx - (N-1).
               # So we update A_array[j - i + N - 1].
               
               # j is 0..N-1.
               # idx is N-1-i .. N-1-i+N-1 = 2N-2-i.
               
               # We need to update this slice of A.
               # But x[j] is the current x (after flip? No, x[j] is unchanged for j!=i).
               # The formula: A_new[k] = A_old[k] - 2 * x_i * x_{i+k}
               # Here x_i is the OLD value?
               # Let's derive:
               # A_old = sum x_n x_{n+k}
               # A_new = sum x'_n x'_{n+k}
               # x'_i = -x_i.
               # Terms involving x_i:
               # 1. n=i: x_i x_{i+k} -> -x_i x_{i+k}. Change: -2 x_i x_{i+k}.
               # 2. n+k=i -> n=i-k: x_{i-k} x_i -> x_{i-k} (-x_i). Change: -2 x_{i-k} x_i.
               # So A_new[k] = A_old[k] - 2 x_i (x_{i+k} + x_{i-k})?
               # Wait, the sum is over n.
               # A_k = sum_{n} x_n x_{n+k}.
               # If we change x_i to -x_i:
               # The term where n=i is x_i x_{i+k}. Becomes -x_i x_{i+k}. Diff: -2 x_i x_{i+k}.
               # The term where n+k=i (i.e., n=i-k) is x_{i-k} x_i. Becomes x_{i-k} (-x_i). Diff: -2 x_{i-k} x_i.
               # So A_new[k] = A_old[k] - 2 x_i x_{i+k} - 2 x_{i-k} x_i?
               # No, these are the same term if we sum over all n?
               # No, the sum is over n.
               # For a fixed k, the pairs are (n, n+k).
               # Only one of n or n+k can be i.
               # If n=i, pair is (i, i+k).
               # If n+k=i, pair is (i-k, i).
               # These are distinct unless k=0.
               # If k=0, pair is (i,i). Term is x_i^2. Becomes x_i^2. No change?
               # x_i^2 = (-x_i)^2.
               # So A_0 does not change!
               # This is consistent with A_0 = sum x_i^2 = N.
               
               # So for k != 0:
               # A_new[k] = A_old[k] - 2 x_i x_{i+k} (if i+k in range)
               # A_new[k] = A_old[k] - 2 x_{i-k} x_i (if i-k in range)
               
               # This is getting complicated to vectorize perfectly without temp arrays.
               # Given N=400, just recomputing A via np.convolve might be faster than complex indexing logic in Python.
               
               x[i] = old_val # Revert temporarily to test cost? No.
               
               # Let's just use np.convolve for simplicity first.
               # If it's too slow, we optimize.
               # 400 iterations * 400 flips = 160,000 convolutions.
               # np.convolve for N=400 is very fast (~10us?).
               # 160,000 * 10us = 1.6 seconds.
               # This is well within 145s.
               
               x[i] = -old_val
               A_new = np.convolve(x, x, mode='full')
               max_new = np.max(np.abs(np.concatenate((A_new[:N-1], A_new[N:]))))
               score_new = max_new / (S_new * S_new)
               
               if score_new < current_score:
                   A = A_new
                   S = S_new
                   current_score = score_new
                   improved = True
               else:
                   x[i] = old_val
       
       return x
   