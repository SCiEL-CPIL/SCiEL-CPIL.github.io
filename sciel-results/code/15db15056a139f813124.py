                    cd = crowding_distance(R_F, front)
                    # cd is aligned with 'front' indices order
                    # We want to pick 'remaining' with highest cd
                    # Sort front indices by their cd value descending
                    # front is a list of indices into R_F
                    # cd is an array of same length as front
                    # Create pairs (index_in_R_F, cd_value)
                    pairs = list(zip(front, cd))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    next_P_indices.extend([p[0] for p in pairs[:remaining]])
