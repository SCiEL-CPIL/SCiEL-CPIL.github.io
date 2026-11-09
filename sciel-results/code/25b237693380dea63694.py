# EVOLVE-BLOCK-START
import numpy as np

def construct():
    p = 1009
    
    # 1. Precompute roots for each non-zero quadratic residue
    # roots[v] = x such that x^2 % p == v
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v != 0 and v not in roots:
            roots[v] = x
            
    # 2. Create a boolean array indicating which values are non-zero QRs
    # is_qr[i] is True if i is a non-zero quadratic residue mod p
    is_qr = np.zeros(p, dtype=bool)
    for v in roots:
        is_qr[v] = True
        
    # 3. Find the smallest s such that s, s+1, s+2, s+3, s+4 are all QRs
    # We need s to be in [1, p-5] because s+4 <= p-1 (since we only care about residues 1..p-1)
    # Note: The problem states 1 <= s <= 1004. Since p=1009, p-5 = 1004.
    
    # To check if i, i+1, i+2, i+3, i+4 are all True:
    # We can use np.roll to shift the array.
    # np.roll(is_qr, k) shifts elements to the right by k.
    # So is_qr[i] & is_qr[i+1] & ... & is_qr[i+4]
    # is equivalent to is_qr & np.roll(is_qr, -1) & np.roll(is_qr, -2) & np.roll(is_qr, -3) & np.roll(is_qr, -4)
    # Wait, np.roll shifts to the right.
    # If we want to check if index i, i+1, i+2, i+3, i+4 are all True:
    # Let A = is_qr.
    # We want A[i] AND A[i+1] AND A[i+2] AND A[i+3] AND A[i+4].
    # This can be computed as:
    # A & np.roll(A, -1) & np.roll(A, -2) & np.roll(A, -3) & np.roll(A, -4)
    # Because np.roll(A, -1)[i] = A[i+1].
    
    # Let's verify:
    # roll(A, -1) shifts elements to the left. So index i gets the value from i+1.
    # Yes, np.roll(A, -k)[i] = A[i+k].
    
    mask = is_qr & np.roll(is_qr, -1) & np.roll(is_qr, -2) & np.roll(is_qr, -3) & np.roll(is_qr, -4)
    
    # We need to find the first True in mask, but with index s >= 1.
    # Also, the indices s+4 must be < p (i.e., <= p-1).
    # Since is_qr[p-1] is the last valid index, and we are checking up to s+4,
    # if s = p-5, we check p-1.
    # The roll operation wraps around. We need to be careful about wrap-around.
    # The problem says "zero residues and wraparound are excluded".
    # This implies we should not consider s where s+i >= p or s+i == 0 (mod p).
    # Since we are looking for residues in 1..p-1, and s >= 1, s+i >= 1.
    # We must ensure s+4 < p. So s <= p-5 = 1004.
    # The roll operation might introduce True values at the end of the array due to wrap-around
    # from the beginning. For example, if is_qr[0] (which is False) was involved, it's fine.
    # But if is_qr[p-1] is True and is_qr[0] is False, roll(is_qr, -1)[p-1] = is_qr[0] = False.
    # However, if we look at index p-5, we check p-5, p-4, p-3, p-2, p-1.
    # If we look at index p-4, we check p-4, p-3, p-2, p-1, p.
    # is_qr[p] is out of bounds, but np.roll wraps to is_qr[0].
    # is_qr[0] is False (0 is not a non-zero QR).
    # So mask[p-4] will be False because of the wrap-around to 0.
    # Similarly for p-3, p-2, p-1.
    # So the wrap-around naturally invalidates the last few indices because 0 is not a QR.
    
    # Find the first True index in mask
    # We only care about indices 1 to p-5.
    # np.where(mask)[0] gives all indices where mask is True.
    true_indices = np.where(mask)[0]
    
    # Filter for valid s: 1 <= s <= p-5
    valid_indices = [i for i in true_indices if 1 <= i <= p - 5]
    
    if not valid_indices:
        # Fallback: should not happen
        raise RuntimeError("No run of 5 consecutive non-zero QRs found")
        
    best_s = min(valid_indices)
    
    # Get the roots
    result_roots = [roots[best_s + i] for i in range(5)]
    
    return {'start': best_s, 'roots': result_roots}
# EVOLVE-BLOCK-END
