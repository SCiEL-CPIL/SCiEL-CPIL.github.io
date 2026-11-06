"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use all multiples of prime 11 up to 128: 11, 22, ..., 121 (11 moduli).
    # Assign distinct residues mod 11 (0..10), canonical 0<=r<m.
    # For any pair (m_a, m_b) = (11a, 11b), gcd = 11*gcd(a,b), which is a multiple of 11.
    # r_a - r_b = a - b, with |a-b| <= 10, so 11 does not divide (r_a - r_b).
    # Hence gcd does not divide (r_a - r_b), so the residue classes are disjoint.
    # This gives 11 classes, better than the seed's 7.
    moduli = [11 * k for k in range(1, 12)]  # 11, 22, ..., 121
    residues = [i for i in range(11)]       # 0, 1, ..., 10 (all < 11 <= modulus)
    return [[residues[i], moduli[i]] for i in range(11)]
# EVOLVE-BLOCK-END
