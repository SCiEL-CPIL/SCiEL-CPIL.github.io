<<<<<<< SEARCH
    # Add multiple Langevin-style self-assembly seeds with different randomizations
    try:
        lang1 = _generate_langevin_seed(size, 99)
        seeds.append(lang1)
    except Exception:
        pass
    
    try:
        lang2 = _generate_langevin_seed(size, 199)
        seeds.append(lang2)
    except Exception:
        pass
=======
    # Add multiple Langevin-style self-assembly seeds with different randomizations
    # These now include periodic relaxation during growth
    try:
        lang1 = _generate_langevin_seed(size, 99, potential, parameter)
        seeds.append(lang1)
    except Exception:
        pass
    
    try:
        lang2 = _generate_langevin_seed(size, 199, potential, parameter)
        seeds.append(lang2)
    except Exception:
        pass
    
    try:
        lang3 = _generate_langevin_seed(size, 299, potential, parameter)
        seeds.append(lang3)
    except Exception:
        pass
>>>>>>> REPLACE

