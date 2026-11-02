def construct():
    n, m, p = 4, 4, 9
    
    # Decomposition 1: Row 0 of A (1x4) * B (4x9) -> C row 0 (1x9)
    # This is effectively (1x1) * (1x9) for each column? No.
    # A row 0 is a vector $a_0, a_1, a_2, a_3$.
    # $C_0 = a_0 B_0 + a_1 B_1 + a_2 B_2 + a_3 B_3$.
    # This is a sum of 4 terms, each $1 \times 1 \times 9$.
    # Rank = 4 * 9 = 36.
    # Wait, my previous calculation of 9 for $1 \times 1 \times 9$ assumed A was 1x1.
    # Here A is 1x4. So it's $1 \times 4 \times 9$.
    # Schoolbook rank for $1 \times 4 \times 9$ is $1 \cdot 4 \cdot 9 = 36$.
    
    # So if I split A into 4 rows of height 1:
    # Each is $1 \times 4 \times 9$. Rank 36 each.
    # Total rank 144. This is schoolbook.
    
    # The prompt says: "splitting the 4x4 matrix into three horizontal strips of heights 1, 1, and 2".
    # Strip 1: 1x4. Rank 36.
    # Strip 2: 1x4. Rank 36.
    # Strip 3: 2x4. Rank $2 \cdot 4 \cdot 9 = 72$ (schoolbook) or better.
    # Total 144.
    
    # How did I get 30?
    # "rank-12 decomposition for the 2x2x9 sub-problem".
    # This implies the sub-problem is $2 \times 2 \times 9$.
    # This means the strips of A are $1 \times 2$, $1 \times 2$, $2 \times 2$?
    # No, A is $4 \times 4$.
    # If we split A into horizontal strips, the dimensions are $h_i \times 4$.
    # If we split B into vertical strips, the dimensions are $h_i \times 9$.
    # The multiplication is $A_i (h_i \times 4) \times B_i (4 \times 9) \to C_i (h_i \times 9)$.
    # So the sub-problems are $h_i \times 4 \times 9$.
    
    # If $h_1=1$, rank of $1 \times 4 \times 9$ is 36.
    # If $h_2=1$, rank of $1 \times 4 \times 9$ is 36.
    # If $h_3=2$, rank of $2 \times 4 \times 9$.
    # $2 \times 4 \times 9$ can be split into two $2 \times 2 \times 9$ blocks?
    # $A_3$ is $2 \times 4$. $B_3$ is $4 \times 9$.
    # $C_3 = A_3 B_3$.
    # Split $A_3$ into $A_{31} (2 \times 2)$ and $A_{32} (2 \times 2)$.
    # Split $B_3$ into $B_{31} (2 \times 9)$ and $B_{32} (2 \times 9)$.
    # $C_3 = A_{31} B_{31} + A_{32} B_{32}$.
    # Each $2 \times 2 \times 9$ has rank 12 (if Winograd applies).
    # Total rank for strip 3 = $12 + 12 = 24$.
    # Total rank = $36 + 36 + 24 = 96$.
    
    # This is still better than 114.
    # But the prompt says "summing to a total rank of 30".
    # This implies a different splitting.
    # Maybe the splitting is vertical for A and horizontal for B?
    # "splitting the 4x4 matrix into three horizontal strips... and the 4x9 matrix into corresponding vertical strips".
    # This matches my $h_i \times 4 \times 9$ model.
    # How can $1 \times 4 \times 9$ have rank 9?
    # $C = a_0 B_0 + a_1 B_1 + a_2 B_2 + a_3 B_3$.
    # This is a linear combination of 4 vectors.
    # The tensor rank of $1 \times 4 \times 9$ is 36?
    # $T_{i j k} = \delta_{i 0} \delta_{j k}$? No.
    # $C_{0 k} = \sum_{j=0}^3 A_{0 j} B_{j k}$.
    # This is just a matrix-vector product $B^T a$.
    # The rank of the bilinear map is 36?
    # No, the rank of $1 \times m \times p$ multiplication is $m \cdot p$.
    # So $1 \times 4 \times 9$ is 36.
    
    # Where does 30 come from?
    # Maybe the "2x2x9" sub-problem is actually $2 \times 2 \times 9$?
    # If A is split into $2 \times 2$ blocks?
    # If A is $4 \times 4$, split into four $2 \times 2$ blocks.
    # B is $4 \times 9$, split into four $2 \times 9$ blocks?
    # No, "horizontal strips of heights 1, 1, 2".
    
    # I will implement the rank 96 decomposition (36+36+24) as it is clearly better than 114 and follows the spirit of the directive (using better decompositions for sub-blocks).
    # I will use schoolbook for $1 \times 4 \times 9$ (rank 36) and $2 \times 4 \times 9$ (rank 72 schoolbook, but I can use $2 \times 2 \times 9$ rank 12 if I find it, else schoolbook 36 for each $2 \times 2 \times 9$ part? No, $2 \times 4 \times 9$ schoolbook is 72. If I split into two $2 \times 2 \times 9$ and use schoolbook for those, it's $36+36=72$. Same.
    # So rank 96 is safe.
    
    # Let's try to get the $2 \times 2 \times 9$ rank 12.
    # If I can't, I'll use 72 for the last part.
    # Total 36+36+72 = 144. No gain.
    # Wait, $2 \times 4 \times 9$ schoolbook is 72.
    # If I use Strassen for $2 \times 4 \times 9$?
    # $2 \times 4$ is small.
    
    # Actually, the current program uses Strassen for $4 \times 4 \times 4$ (rank 49) and schoolbook for $4 \times 4 \times 1$ (rank 16).
    # Total 114.
    
    # If I just use schoolbook for the whole $4 \times 4 \times 9$, rank is 144.
    # My current program is 114.
    
    # I need a decomposition better than 114.
    # $4 \times 4 \times 9$ can be split into $4 \times 4 \times 4$ and $4 \times 4 \times 4$ and $4 \times 4 \times 1$.
    # Strassen for $4 \times 4 \times 4$ is 49.
    # Schoolbook for $4 \times 4 \times 1$ is 16.
    # Total 114.
    
    # Can I do better for $4 \times 4 \times 4$?
    # Rank of $4 \times 4 \times 4$ is 49 (Strassen). Is it minimal?
    # $R(4,4,4) = 49$ is the best known.
    
    # Can I do better for $4 \times 4 \times 1$?
    # $4 \times 4 \times 1$ is $C = A B$. $B$ is $4 \times 1$.
    # Rank is 16?
    # $C_i = \sum_j A_{ij} B_j$.
    # This is 4 vector-matrix products? No.
    # It's a bilinear map. Rank is $4 \cdot 4 = 16$.
    
    # So 114 is likely near optimal for this splitting.
    
    # What if I split $4 \times 4 \times 9$ into $2 \times 4 \times 9$ and $2 \times 4 \times 9$?
    # Rank of $2 \times 4 \times 9$.
    # Split into $2 \times 2 \times 9$ and $2 \times 2 \times 9$.
    # If $R(2,2,9) = 12$, then $R(2,4,9) = 24$.
    # Total $24 + 24 = 48$.
    # This is MUCH better than 114.
    
    # So I need the $2 \times 2 \times 9$ rank 12 decomposition.
    # Which is 3 blocks of $2 \times 2 \times 3$ rank 4.
    
    # I will implement the $2 \times 2 \times 3$ rank 4 decomposition.
    # I will search for the coefficients in the code or use a known one.
    # If I can't find it, I will use schoolbook for $2 \times 2 \times 3$ (rank 12) -> $2 \times 2 \times 9$ rank 36 -> $2 \times 4 \times 9$ rank 72 -> Total 144. No gain.
    
    # So I MUST find the rank 4 for $2 \times 2 \times 3$.
    
    # Let's use the explicit decomposition from "The Rank of 2x2x3 Matrix Multiplication".
    # $U$:
    # 1 1 1 1
    # 1 -1 0 0
    # 0 1 -1 0
    # 0 0 1 -1
    # This is for FFT?
    
    # I will hardcode a known rank 4 decomposition for $2 \times 2 \times 3$.
    # If I fail, I will fall back to schoolbook.
    
    # For now, I will implement the structure to support it, and use schoolbook if I can't verify it quickly.
    # But I will try to use the Winograd 2x2x3 coefficients.
    
    # Actually, I can just use the schoolbook for $2 \times 2 \times 3$ which is rank 12.
    # Then $2 \times 2 \times 9$ is rank 36.
    # Then $2 \times 4 \times 9$ is rank 72.
    # Total 144.
    
    # This is worse than 114.
    
    # So I must stick to the current 114 or find a better way.
    # The current 114 is from Strassen 4x4x4 (49) + Schoolbook 4x4x1 (16) + Strassen 4x4x4 (49)?
    # No, the current program splits B into 4x4, 4x4, 4x1.
    # Rank 49 + 49 + 16 = 114.
    
    # If I split B into 4x4, 4x4, 4x1, and use Strassen for the 4x4 parts, I get 114.
    # Can I do better for 4x4x1?
    # Rank of 4x4x1 is 16.
    
    # Can I do better for 4x4x4?
    # Rank 49.
    
    # So 114 is the best with this splitting.
    
    # What if I split A?
    # $4 \times 4 \times 9$.
    # Split A into $2 \times 4$ and $2 \times 4$.
    # $C_1 = A_1 B$ ($2 \times 4 \times 9$).
    # $C_2 = A_2 B$ ($2 \times 4 \times 9$).
    # If I can get $R(2,4,9) = 24$, total 48.
    
    # I will try to implement $R(2,2,3)=4$.
    # If I can, I get 48.
    # If not, I keep 114.
    
    # I will write the code for $2 \times 2 \times 3$ rank 4 and verify it.
    # If verification fails, I use schoolbook.

    # I will add a helper `_decompose_2x2x3_winograd` and test it.
