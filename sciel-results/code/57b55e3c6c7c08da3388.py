"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Maximize the number of pairwise disjoint residue classes with distinct moduli in [2, 128].
    # Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does not divide (r1 - r2).
    # A robust strategy is to select moduli that are all multiples of a common integer K.
    # Then gcd(m_i, m_j) is a multiple of K.
    # If we assign distinct residues modulo K, then K does not divide (r_i - r_j).
    # Consequently, gcd(m_i, m_j) does not divide (r_i - r_j).
    # The number of such classes is limited by:
    # 1. The number of multiples of K in [2, 128], which is floor(128/K).
    # 2. The number of distinct residues modulo K, which is K.
    # We want to maximize min(floor(128/K), K).
    # Testing values of K:
    # K=11: floor(128/11)=11, min(11, 11)=11.
    # K=12: floor(128/12)=10, min(10, 12)=10.
    # K=10: floor(128/10)=12, min(12, 10)=10.
    # K=13: floor(128/13)=9, min(9, 13)=9.
    # The maximum is 11, achieved with K=11.
    K = 11
    moduli = [K * k for k in range(1, 129) if K * k <= 128]
    # We have 11 moduli. We assign distinct residues 0..10.
    result = []
    for i, m in enumerate(moduli):
        r = i % K
        result.append([r, m])
    return result
# EVOLVE-BLOCK-END
