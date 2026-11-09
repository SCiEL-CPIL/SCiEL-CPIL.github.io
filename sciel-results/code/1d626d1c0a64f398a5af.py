# Precompute once:
# F_n(ξ) = FT[e^{-x²/4} H_n(x)] for n ∈ {0, 4, 8, 12}
# Then F(ξ) = Σ a_n F_n(ξ)
# |F|² = Σ_{n,m} a_n a_m F_n F_m*  (but F_n are real for even n, so F_n F_m)
# Actually F_n may be complex for odd n, but we only have even n, so they're real
# ∫|F|² dξ = Σ_{n,m} a_n a_m G_{nm} where G_{nm} = ∫ F_n F_m dξ
# ∫ξ²|F|² dξ = Σ_{n,m} a_n a_m H_{nm} where H_{nm} = ∫ ξ² F_n F_m dξ
