import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 5, 5, 5

    # Block sizes: 5 = 3 + 2
    block_sizes = [3, 2]

    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Strassen decomposition for 2x2x2 (rank 7)
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    # Coppersmith's rank-23 algorithm for 3x3 matrix multiplication
    # Each factor is (U, V, W) where:
    # U has 9 entries: U[li*3+lj] = coefficient of a_{li,lj}
    # V has 9 entries: V[lj*3+lk] = coefficient of b_{lj,lk}
    # W has 9 entries: W[li*3+lk] = coefficient of c_{li,lk}
    def get_coppersmith_3x3_factors():
        # I'll use a verified rank-23 decomposition.
        # Let me define the 23 terms and verify numerically.

        # The matmul tensor: T[a_idx, b_idx, c_idx] = 1 if
        # c_{p,q} = sum_r a_{p,r} * b_{r,q}
        # a_idx = p*3+r, b_idx = r*3+q, c_idx = p*3+q

        # I'll use the following 23-term decomposition:
        terms = []

        # Helper to create a vector
        def vec(pairs):
            v = [0]*9
            for idx, val in pairs:
                v[idx] = val
            return v

        # Term 1: (a00+a01+a02+a10+a11+a12+a20+a21+a22)*(b00+b01+b02+b10+b11+b12+b20+b21+b22)
        # -> all c entries get +1
        all_ones = [1]*9
        terms.append((all_ones[:], all_ones[:], all_ones[:]))

        # Term 2: (a00+a01+a02)*(b00+b01+b02)
        # U: a00,a01,a02 -> indices 0,1,2
        # V: b00,b01,b02 -> indices 0,1,2
        # W: c00,c01,c02 -> indices 0,1,2
        terms.append(([1,1,1,0,0,0,0,0,0], [1,1,1,0,0,0,0,0,0], [1,1,1,0,0,0,0,0,0]))

        # Term 3: (a10+a11+a12)*(b10+b11+b12)
        terms.append(([0,0,0,1,1,1,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,1,1,1,0,0,0]))

        # Term 4: (a20+a21+a22)*(b20+b21+b22)
        terms.append(([0,0,0,0,0,0,1,1,1], [0,0,0,0,0,0,1,1,1], [0,0,0,0,0,0,1,1,1]))

        # Term 5: (a00+a10+a20)*(b00+b10+b20)
        # U: a00,a10,a20 -> indices 0,3,6
        # V: b00,b10,b20 -> indices 0,3,6
        # W: c00,c10,c20 -> indices 0,3,6
        terms.append(([1,0,0,1,0,0,1,0,0], [1,0,0,1,0,0,1,0,0], [1,0,0,1,0,0,1,0,0]))

        # Term 6: (a01+a11+a21)*(b01+b11+b21)
        terms.append(([0,1,0,0,1,0,0,1,0], [0,1,0,0,1,0,0,1,0], [0,1,0,0,1,0,0,1,0]))

        # Term 7: (a02+a12+a22)*(b02+b12+b22)
        terms.append(([0,0,1,0,0,1,0,0,1], [0,0,1,0,0,1,0,0,1], [0,0,1,0,0,1,0,0,1]))

        # Term 8: (a00+a11+a22)*(b00+b11+b22)
        terms.append(([1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1], [1,0,0,0,1,0,0,0,1]))

        # Term 9: (a02+a11+a20)*(b02+b11+b20)
        terms.append(([0,0,1,0,1,0,1,0,0], [0,0,1,0,1,0,1,0,0], [0,0,1,0,1,0,1,0,0]))

        # Term 10: (a01+a10+a21)*(b01+b10+b21)
        terms.append(([0,1,0,1,0,0,0,1,0], [0,1,0,1,0,0,0,1,0], [0,1,0,1,0,0,0,1,0]))

        # Term 11: (a00+a01+a10+a11)*(b00+b01+b10+b11)
        terms.append(([1,1,0,1,1,0,0,0,0], [1,1,0,1,1,0,0,0,0], [1,1,0,1,1,0,0,0,0]))

        # Term 12: (a02+a11+a21+a22)*(b02+b11+b21+b22)
        terms.append(([0,0,1,0,1,0,0,1,1], [0,0,1,0,1,0,0,1,1], [0,0,1,0,1,0,0,1,1]))

        # Term 13: (a00+a20+a10+a21)*(b00+b20+b10+b21)
        terms.append(([1,0,0,1,0,0,1,1,0], [1,0,0,1,0,0,1,1,0], [1,0,0,1,0,0,1,1,0]))

        # Term 14: (a01+a22+a11+a12)*(b01+b22+b11+b12)
        terms.append(([0,1,0,0,1,1,0,0,1], [0,1,0,0,1,1,0,0,1], [0,1,0,0,1,1,0,0,1]))

        # Term 15: (a00+a02+a20+a22)*(b00+b02+b20+b22)
        terms.append(([1,0,1,0,0,0,1,0,1], [1,0,1,0,0,0,1,0,1], [1,0,1,0,0,0,1,0,1]))

        # Term 16: (a01+a10+a21)*(b01+b10+b21) -- same as term 10!
        # Need different terms. Let me use:
        # Term 16: (a00+a10+a22)*(b00+b10+b22)
        terms.append(([1,0,0,1,0,0,0,0,1], [1,0,0,1,0,0,0,0,1], [1,0,0,1,0,0,0,0,1]))

        # Term 17: (a02+a11+a20)*(b02+b11+b20) -- same as term 9!
        # Term 17: (a00+a11+a22)*(b00+b11+b22) -- same as term 8!
        # I'm running into duplicates. This means my decomposition is wrong.

        # Let me just use schoolbook for 3x3x3 (rank 27) to be safe,
        # and rely on the 3+2 partition giving rank 27+4+4+1+7+4+4+1 = 52
        # vs the 2+2+1 partition giving 7+4+4+1+4+4+1+4+4+1 = 34... wait.

        # Let me recalculate. For 3+2 partition:
        # (3,3,3): rank 27 (schoolbook) or 23 (Coppersmith)
        # (3,3,2): 3*3*2 = 18
        # (3,2,3): 3*2*3 = 18
        # (3,2,2): 3*2*2 = 12
        # (2,3,3): 2*3*3 = 18
        # (2,3,2): 2*3*2 = 12
        # (2,2,3): 2*2*3 = 12
        # (2,2,2): 7 (Strassen)
        # Total with schoolbook 3x3: 27+18+18+12+18+12+12+7 = 124 (worse than 117!)
        # Total with Coppersmith 3x3: 23+18+18+12+18+12+12+7 = 120 (still worse than 117!)

        # So the 3+2 partition is WORSE than 2+2+1! Let me stick with 2+2+1.

        # For 2+2+1 partition:
        # (2,2,2): 7
        # (2,2,1): 4
        # (2,1,2): 4
        # (2,1,1): 2
        # (1,2,2): 4
        # (1,2,1): 2
        # (1,1,2): 2
        # (1,1,1): 1
        # Total: 7+4+4+2+4+2+2+1 = 26... that's way less than 117!

        # Wait, I think I'm confusing myself. Let me recalculate.
        # In the 2+2+1 partition, we have blocks of sizes 2, 2, 1.
        # The block (i,j,k) has size ni x mj x pk where ni, mj, pk are from {2,2,1}.
        #
        # (2,2,2): 7
        # (2,2,1): 2*2*1 = 4
        # (2,1,2): 2*1*2 = 4
        # (2,1,1): 2*1*1 = 2
        # (1,2,2): 1*2*2 = 4
        # (1,2,1): 1*2*1 = 2
        # (1,1,2): 1*1*2 = 2
        # (1,1,1): 1*1*1 = 1
        # Total: 7+4+4+2+4+2+2+1 = 26

        # But the current program says rank 117! Let me re-examine.
        #
        # OH WAIT. I think the issue is that in the block decomposition,
        # the "blocks" are not independent. When we partition A into blocks
        # A[i,j], B into B[j,k], C into C[i,k], the multiplication of
        # block (i,j,k) means: C[i,k] += A[i,j] * B[j,k]
        #
        # So for each (i,j,k) triple, we need to multiply an ni x mj matrix
        # by an mj x pk matrix to get an ni x pk matrix. This requires
        # ni*mj*pk scalar multiplications (schoolbook) or fewer (Strassen).
        #
        # So the total rank is sum over all (i,j,k) of rank(ni x mj x pk).
        #
        # For 2+2+1:
        # (2,2,2): 7
        # (2,2,1): 2*2*1 = 4
        # (2,1,2): 2*1*2 = 4
        # (2,1,1): 2*1*1 = 2
        # (1,2,2): 1*2*2 = 4
        # (1,2,1): 1*2*1 = 2
        # (1,1,2): 1*1*2 = 2
        # (1,1,1): 1*1*1 = 1
        # Total: 26

        # But the current program has rank 117! Let me look at the code again.
        #
        # Ah, I see. The current program uses block_sizes = [2, 2, 1] and
        # iterates i, j, k over range(3). So there are 3x3x3 = 27 block
        # combinations. But wait, the block sizes are [2,2,1], so:
        # i=0: ni=2, i=1: ni=2, i=2: ni=1
        # j=0: mj=2, j=1: mj=2, j=2: mj=1
        # k=0: pk=2, k=1: pk=2, k=2: pk=1
        #
        # So the 27 combinations are:
        # (2,2,2): 7
        # (2,2,2): 7  -- wait, k=1 also has pk=2!
        #
        # OH! I see the issue. The block sizes are [2,2,1], so:
        # k=0: pk=2, k=1: pk=2, k=2: pk=1
        #
        # So (i,j,k) combinations:
        # (0,0,0): ni=2,mj=2,pk=2 -> 7
        # (0,0,1): ni=2,mj=2,pk=2 -> 7
        # (0,0,2): ni=2,mj=2,pk=1 -> 4
        # (0,1,0): ni=2,mj=2,pk=2 -> 7
        # (0,1,1): ni=2,mj=2,pk=2 -> 7
        # (0,1,2): ni=2,mj=2,pk=1 -> 4
        # (0,2,0): ni=2,mj=1,pk=2 -> 4
        # (0,2,1): ni=2,mj=1,pk=2 -> 4
        # (0,2,2): ni=2,mj=1,pk=1 -> 2
        # (1,0,0): ni=2,mj=2,pk=2 -> 7
        # (1,0,1): ni=2,mj=2,pk=2 -> 7
        # (1,0,2): ni=2,mj=2,pk=1 -> 4
        # (1,1,0): ni=2,mj=2,pk=2 -> 7
        # (1,1,1): ni=2,mj=2,pk=2 -> 7
        # (1,1,2): ni=2,mj=2,pk=1 -> 4
        # (1,2,0): ni=2,mj=1,pk=2 -> 4
        # (1,2,1): ni=2,mj=1,pk=2 -> 4
        # (1,2,2): ni=2,mj=1,pk=1 -> 2
        # (2,0,0): ni=1,mj=2,pk=2 -> 4
        # (2,0,1): ni=1,mj=2,pk=2 -> 4
        # (2,0,2): ni=1,mj=2,pk=1 -> 2
        # (2,1,0): ni=1,mj=2,pk=2 -> 4
        # (2,1,1): ni=1,mj=2,pk=2 -> 4
        # (2,1,2): ni=1,mj=2,pk=1 -> 2
        # (2,2,0): ni=1,mj=1,pk=2 -> 2
        # (2,2,1): ni=1,mj=1,pk=2 -> 2
        # (2,2,2): ni=1,mj=1,pk=1 -> 1
        #
        # Total: 7*8 + 4*8 + 4*4 + 2*4 + 2*2 + 1 = 56+32+16+8+4+1 = 117. Yes!

        # So the 2+2+1 partition has 8 blocks of size 2x2x2 (since there are
        # two "2" blocks in each dimension), giving 8*7 = 56 from Strassen.
        # The remaining 19 blocks use schoolbook.
        #
        # Now, can we do better? What about using a different partition?
        #
        # What about 5 = 3 + 2? Then we have 2x2x2 = 8 blocks:
        # (3,3,3): 27 (schoolbook) or 23 (Coppersmith)
        # (3,3,2): 18
        # (3,2,3): 18
        # (3,2,2): 12
        # (2,3,3): 18
        # (2,3,2): 12
        # (2,2,3): 12
        # (2,2,2): 7
        # Total with schoolbook: 27+18+18+12+18+12+12+7 = 124
        # Total with Coppersmith: 23+18+18+12+18+12+12+7 = 120
        # Both worse than 117!
        #
        # What about 5 = 1 + 2 + 2 (same as 2+2+1, just reordered)?
        # Same result: 117.
        #
        # What about 5 = 4 + 1?
        # (4,4,4): 64 (schoolbook)
        # (4,4,1): 16
        # (4,1,4): 16
        # (4,1,1): 4
        # (1,4,4): 16
        # (1,4,1): 4
        # (1,1,4): 4
        # (1,1,1): 1
        # Total: 64+16+16+4+16+4+4+1 = 125. Worse!
        #
        # What about 5 = 2 + 3?
        # Same as 3+2: 124 or 120. Worse.
        #
        # So 2+2+1 with Strassen on 2x2x2 blocks giving 117 is the best
        # among simple block decompositions.
        #
        # Can we do better than 117? The theoretical lower bound for 5x5
        # matrix multiplication is at least 5^2 = 25 (trivial) and at most
        # 125 (schoolbook). The best known upper bound is much lower.
        #
        # Actually, I recall that the best known rank for 5x5 matrix
        # multiplication is around 103-110 or so. But I don't have a
        # specific algorithm in mind.
        #
        # For now, let me stick with the 2+2+1 partition giving rank 117,
        # which is what the current program does. The inspirations are
        # essentially the same program.
        #
        # Let me just return the schoolbook factors for 3x3x3 (rank 27)
        # in case we use the 3+2 partition, but we won't since it's worse.

        # For this function, I'll just return schoolbook 3x3x3 (rank 27)
        # since we're not using it in the final program.
        schoolbook = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U = [0]*9
                    V = [0]*9
                    W = [0]*9
                    U[i*3+j] = 1
                    V[j*3+k] = 1
                    W[i*3+k] = 1
                    schoolbook.append((U, V, W))
        return schoolbook

    strassen_factors = get_strassen_factors()

    # Use 2+2+1 partition (best known simple decomposition)
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Total rank calculation
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    pass

                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]

                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1)
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1)
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1)
                        ]

                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx], col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx], col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx], col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx, col] = 1.0
                                V[V_idx, col] = 1.0
                                W[W_idx, col] = 1.0
                                col += 1

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
