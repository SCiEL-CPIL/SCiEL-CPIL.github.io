    rank = R4  # 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((32, rank), dtype=np.float64)
    W = np.zeros((32, rank), dtype=np.float64)

    U[:] = U4
    
    for j in range(4):
        for k in range(4):
            # Left half (k=0..3)
            V[j * 8 + k, :] = V4[j * 4 + k, :]
            W[k * 4 + j, :] = W4[k * 4 + j, :]
            
            # Right half (k=4..7)
            # Note: W index is k*n+i. Here j is used as i in W4?
            # W4 shape is (16, 49). W4[k*4+i, r].
            # In the loop, j is the column of A (which is i in C? No, j is col of A, i is row of A).
            # In W, index is k*n+i.
            # In W4, index is k*4+i.
            # So W4[k*4+i, r] corresponds to W[k*4+i, r].
            # In the loop variable `j` is used for A's column.
            # In the code `W[k * 4 + j, ...]`, `j` is being used as the `i` index for W?
            # Let's check the previous code: `W[k * 4 + j, 0:R4] = W4[k * 4 + j, :]`.
            # Yes, the loop variable `j` iterates over 0..3.
            # In the context of `W[k*n+i]`, `i` is the row index of C (and A).
            # In the context of `V[j*p+k]`, `j` is the column index of A (and row index of B).
            # So `j` in the loop is the shared index between A's col and B's row.
            # But `W` depends on `i` (A's row) and `k` (B's col).
            # The loop is `for j in range(4): for k in range(4):`.
            # This `j` is A's col.
            # But `W` needs `i` (A's row).
            # Wait, the previous code `W[k * 4 + j, ...]` uses `j` as the `i` index?
            # That would be wrong if `j` is A's col.
            # Let's re-read the Strassen construction in the code.
            
            # U4[i*4+j, r]. i=row A, j=col A.
            # V4[j*4+k, r]. j=row B (col A), k=col B.
            # W4[k*4+i, r]. k=col B (row C), i=row A (col C).
            
            # In the assembly loop:
            # for j in range(4): # This j is col A / row B
            #   for k in range(4): # This k is col B
            #     V[j*8+k, ...] = V4[j*4+k, ...] # Correct. j is row B, k is col B.
            #     W[k*4+j, ...] = W4[k*4+j, ...] # Incorrect?
            #     W index is k*4 + i.
            #     The code uses `j` as the second index.
            #     But `j` is col A. `i` is row A.
            #     So `W[k*4+j]` implies `i=j`.
            #     This is only correct if we are iterating over `i` as well?
            #     No, the loop is only over `j` and `k`.
            #     This implies that the previous code might be WRONG?
            #     Or my interpretation of the indices is wrong.
            
            # Let's check the verification in the current code.
            # It passed with score 1.3061.
            # So the decomposition must be valid.
            # How can `W[k*4+j]` be correct if `j` is col A and `i` is row A?
            # Unless... the loop `for j in range(4)` is actually iterating over `i`?
            # No, `V[j*8+k]` uses `j` as row B. Row B is Col A.
            # So `j` is Col A.
            # `W[k*4+i]` uses `i` as Row A.
            # If the code sets `W[k*4+j]`, it is setting `i=j`.
            # This would mean the decomposition only works if `i=j`? No, that's not a matrix multiplication.
            
            # Let's look closely at the Strassen block construction.
            # U4 is built correctly.
            # V4 is built correctly.
            # W4 is built correctly.
            
            # Let's look at the assembly again.
            # `for j in range(4): for k in range(4):`
            # `V[j * 8 + k, 0:R4] = V4[j * 4 + k, :]`
            # `W[k * 4 + j, 0:R4] = W4[k * 4 + j, :]`
            
            # If this code is correct, then `j` in `W[k*4+j]` MUST correspond to `i` in `W4[k*4+i]`.
            # But `j` in `V[j*4+k]` MUST correspond to `j` in `V4[j*4+k]`.
            # In `V4`, the first index is Row B (Col A).
            # In `W4`, the second index is Row A.
            # So the loop variable `j` is being used as BOTH Col A and Row A?
            # This is impossible for a general matrix.
            
            # UNLESS... the "j" in the loop is not Col A.
            # Let's check the variable names in the loop.
            # `for j in range(4):`
            # It's just a variable name.
            # Maybe the previous code is wrong but the verification passed?
            # No, verification uses `einsum`.
            # If the assembly was wrong, verification would fail.
            # So the assembly MUST be correct.
            # How?
            
            # Ah! `W` shape is `(32, 98)`. `W[k*n+i, r]`.
            # `W4` shape is `(16, 49)`. `W4[k*4+i, r]`.
            # The code says: `W[k * 4 + j, 0:R4] = W4[k * 4 + j, :]`.
            # This assigns `W[index] = W4[index]`.
            # The index is `k*4+j`.
            # This covers all indices `0..15` for a fixed `k`?
            # No, `k` goes 0..3, `j` goes 0..3.
            # So `k*4+j` covers 0..15.
            # So it fills the entire `W` block for the left half.
            # It just uses the variable `j` to iterate over the `i` dimension of `W`.
            # And the variable `j` to iterate over the `j` dimension of `V`.
            # So the loop variable `j` is serving double duty in the index calculation,
            # but since `V` and `W` are separate matrices, and we are filling them independently,
            # we just need to ensure that for `V`, `j` is the correct row (Col A),
            # and for `W`, `j` is the correct col (Row A).
            
            # Is it valid to use the same loop variable `j` for both?
            # Yes, because we are just iterating over 0..3 for both dimensions.
            # The value of `j` at any step is the same for both `V` and `W` assignments.
            # For `V`, we need to fill `V[row_B, col_B]`. `row_B` = `j`.
            # For `W`, we need to fill `W[col_B, row_A]`. `row_A` = `j`.
            # This implies that we are filling `V[j, k]` and `W[k, j]` simultaneously.
            # This is fine, as long as `V` and `W` are defined correctly in `V4` and `W4`.
            # `V4` is indexed by `row_B, col_B`.
            # `W4` is indexed by `col_B, row_A`.
            # So `V4[j, k]` and `W4[k, j]` are the correct entries to fetch.
            # And we store them in `V[j*8+k]` and `W[k*4+j]`.
            # This is perfectly correct.
            # The fact that `j` represents Col A in `V` and Row A in `W` is just a coincidence of the loop variable name,
            # but since we iterate over all 0..3 for both, we cover all necessary entries.
            
            # So the logic holds.
            # My proposed optimization:
            # Use rank 49.
            # `U[:] = U4`
            # `V[j*8+k, :] = V4[j*4+k, :]`
            # `W[k*4+j, :] = W4[k*4+j, :]`
            # And for the right half:
            # `V[j*8+(4+k), :] = V4[j*4+k, :]`
            # `W[(4+k)*4+j, :] = W4[k*4+j, :]`
            
            # This fills `V` and `W` for all `k` (0..7) using the same `U4`.
            # This is valid.

    # Verification
    # The verification code uses `rank` variable.
    # I need to update `rank` to 49.
    # And the slicing in verification `U.reshape(4, 4, rank)` etc will work.

    # One detail: `U` in the new scheme is `U4`.
    # `U4` is `(16, 49)`.
    # `U` is `(16, 49)`.
    # `V` is `(32, 49)`.
    # `W` is `(32, 49)`.

    # The verification code:
    # `U_3d = U.reshape(4, 4, rank)`
    # `V_3d = V.reshape(4, 8, rank)`
    # `W_3d = W.reshape(8, 4, rank)`
    # This matches the shapes.

    # So the change is simple.
